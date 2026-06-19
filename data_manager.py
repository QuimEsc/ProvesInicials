import os
import re
import time
import random
import threading
import urllib.request
from typing import Dict, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Manté exactament el mateix nom de carpeta
CARPETA_DADES = os.environ.get("DADES_FOLDER", os.path.join(BASE_DIR, "dades_historiques"))

REFRESH_INTERVAL_MINUTES = 30

# Nombre de dies recents que es tornen a descarregar i sobreescriure.
RECENT_REFRESH_DAYS = 5

# Configuració de descàrrega
FULL_DOWNLOAD_PERIOD = "max"
YF_INTERVAL = "1d"
YF_AUTO_ADJUST = False
YF_REPAIR = False

# Nou: control suau de peticions per reduir risc de bloqueig
YF_TIMEOUT_SECONDS = float(os.environ.get("YF_TIMEOUT_SECONDS", "20"))
YF_MAX_RETRIES = int(os.environ.get("YF_MAX_RETRIES", "4"))
YF_BASE_BACKOFF_SECONDS = float(os.environ.get("YF_BASE_BACKOFF_SECONDS", "2.0"))
YF_RATE_LIMIT_BACKOFF_SECONDS = float(os.environ.get("YF_RATE_LIMIT_BACKOFF_SECONDS", "20.0"))
YF_MIN_SECONDS_BETWEEN_REQUESTS = float(os.environ.get("YF_MIN_SECONDS_BETWEEN_REQUESTS", "1.2"))
YF_RANDOM_JITTER_SECONDS = float(os.environ.get("YF_RANDOM_JITTER_SECONDS", "0.8"))
FRED_TIMEOUT_SECONDS = float(os.environ.get("FRED_TIMEOUT_SECONDS", "12"))
FRED_ECBDFR_URL = os.environ.get(
    "FRED_ECBDFR_URL",
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBDFR",
)
FRED_ECBDFR_TEXT_URL = os.environ.get(
    "FRED_ECBDFR_TEXT_URL",
    "https://fred.stlouisfed.org/data/ECBDFR",
)
FRED_SERIES_URLS = {
    "DFEDTARU": os.environ.get(
        "FRED_DFEDTARU_URL",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU",
    ),
    "DFEDTARL": os.environ.get(
        "FRED_DFEDTARL_URL",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARL",
    ),
    "DFF": os.environ.get(
        "FRED_DFF_URL",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF",
    ),
    "FEDFUNDS": os.environ.get(
        "FRED_FEDFUNDS_URL",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
    ),
}
FEDERAL_RESERVE_DFF_URL = os.environ.get(
    "FEDERAL_RESERVE_DFF_URL",
    "https://www.federalreserve.gov/datadownload/Output.aspx"
    "?rel=H15&series=646250c87b1afd04cc6774796fc0cec8&lastObs=&from=&to="
    "&filetype=csv&label=include&layout=seriescolumn",
)
ECB_DFR_URL = os.environ.get(
    "ECB_DFR_URL",
    "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV?format=csvdata",
)

# Logs en consola
ENABLE_LOGS = str(os.environ.get("ENABLE_DATA_MANAGER_LOGS", "1")).strip().lower() not in {"0", "false", "no", "off"}

os.makedirs(CARPETA_DADES, exist_ok=True)

_CACHE_MEMORIA: Dict[str, Tuple[pd.DataFrame, pd.Timestamp]] = {}

# Nou: throttle global simple per a les peticions HTTP
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0


def _log(msg: str) -> None:
    if ENABLE_LOGS:
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[data_manager {now}] {msg}")


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def _safe_filename(ticker: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(ticker))


def _csv_path(ticker: str) -> str:
    return os.path.join(CARPETA_DADES, f"{_safe_filename(ticker)}.csv")


def _rate_csv_path(series_id: str) -> str:
    return os.path.join(CARPETA_DADES, f"{_safe_filename(series_id)}.csv")


def _utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None)


def _empty_ohlcv_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex([], name="Date")
    return df


def _file_timestamp_utc(path: str) -> pd.Timestamp | None:
    if not os.path.exists(path):
        return None
    return pd.Timestamp.fromtimestamp(os.path.getmtime(path), tz="UTC").tz_localize(None)


def _is_timestamp_fresh(ts: pd.Timestamp | None) -> bool:
    if ts is None:
        return False
    return (_utc_now_naive() - ts) < pd.Timedelta(minutes=REFRESH_INTERVAL_MINUTES)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out.index = pd.DatetimeIndex([], name="Date")
        return out

    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")

    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    out.index = pd.DatetimeIndex(idx)
    out = out[~out.index.isna()].copy()
    out = out.sort_index(kind="mergesort")

    # En dades diàries volem una sola fila per data.
    # Si hi ha dos timestamps del mateix dia, ens quedem amb l'últim.
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].copy()

    out.index.name = "Date"
    return out.sort_index(kind="mergesort")


def _ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_ohlcv_df()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    rename_map = {}
    for col in out.columns:
        low = str(col).strip().lower()
        if low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low == "volume":
            rename_map[col] = "Volume"

    if rename_map:
        out = out.rename(columns=rename_map)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[["Open", "High", "Low", "Close", "Volume"]].copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return _ensure_datetime_index(out)


def _fill_synthetic_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_ohlcv_schema(df)

    if out.empty:
        return out

    out = out.dropna(subset=["High", "Low"]).copy()

    if out.empty:
        return _empty_ohlcv_df()

    midpoint = (out["High"] + out["Low"]) / 2.0
    out["Close"] = out["Close"].fillna(midpoint)
    out["Open"] = out["Open"].fillna(out["Close"].shift(1))
    out["Open"] = out["Open"].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(1.0)

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    out.index.name = "Date"
    return out


def _merge_frames_by_date(*dfs: pd.DataFrame) -> pd.DataFrame:
    valid = []
    for df in dfs:
        if df is None:
            continue
        cleaned = _ensure_ohlcv_schema(df)
        if not cleaned.empty:
            valid.append(cleaned)

    if not valid:
        return _empty_ohlcv_df()

    merged = pd.concat(valid, axis=0)
    merged = _ensure_datetime_index(merged)
    merged = _fill_synthetic_ohlcv(merged)
    return merged


def _last_candle_signature(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    last_idx = df.index[-1]
    row = df.iloc[-1]

    return (
        pd.Timestamp(last_idx).strftime("%Y-%m-%d"),
        None if pd.isna(row.get("Open")) else float(row["Open"]),
        None if pd.isna(row.get("High")) else float(row["High"]),
        None if pd.isna(row.get("Low")) else float(row["Low"]),
        None if pd.isna(row.get("Close")) else float(row["Close"]),
        None if pd.isna(row.get("Volume")) else float(row["Volume"]),
    )


def _wait_before_request() -> None:
    global _LAST_REQUEST_TS

    with _REQUEST_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_REQUEST_TS
        wait_needed = max(0.0, YF_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

        if wait_needed > 0:
            time.sleep(wait_needed)

        jitter = random.uniform(0.0, max(0.0, YF_RANDOM_JITTER_SECONDS))
        if jitter > 0:
            time.sleep(jitter)

        _LAST_REQUEST_TS = time.monotonic()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    patterns = [
        "429",
        "too many requests",
        "rate limit",
        "rate-limited",
        "ratelimit",
        "temporarily blocked",
        "forbidden",
    ]
    return any(p in msg for p in patterns)


def _download_from_yahoo_once(
    ticker: str,
    *,
    start: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("Falta yfinance. Instal·la yfinance o puja els CSV manuals.")

    _wait_before_request()

    # Mantinc una sola descàrrega per ticker, sense paral·lelisme
    # per ser més amable amb el servei.
    kwargs = {
        "tickers": ticker,
        "interval": YF_INTERVAL,
        "auto_adjust": YF_AUTO_ADJUST,
        "repair": YF_REPAIR,
        "progress": False,
        "threads": False,
        "timeout": YF_TIMEOUT_SECONDS,
    }

    if start is not None:
        kwargs["start"] = start
    else:
        kwargs["period"] = period or FULL_DOWNLOAD_PERIOD

    df = yf.download(**kwargs)

    if df is None or df.empty:
        return _empty_ohlcv_df()

    df = _ensure_ohlcv_schema(df)
    df = _fill_synthetic_ohlcv(df)
    return df


def _download_from_yahoo(
    ticker: str,
    *,
    start: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    last_exc = None

    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            return _download_from_yahoo_once(ticker, start=start, period=period)

        except KeyboardInterrupt:  # pragma: no cover
            raise
        except Exception as exc:
            last_exc = exc

            if attempt >= YF_MAX_RETRIES:
                break

            if _is_rate_limit_error(exc):
                sleep_s = YF_RATE_LIMIT_BACKOFF_SECONDS * attempt + random.uniform(0.0, YF_RANDOM_JITTER_SECONDS)
            else:
                sleep_s = YF_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(
                    0.0, YF_RANDOM_JITTER_SECONDS
                )

            _log(
                f"{ticker}: error en descàrrega (intent {attempt}/{YF_MAX_RETRIES}): {exc}. "
                f"Reintentant en {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"No s'han pogut descarregar dades de Yahoo per a {ticker}: {last_exc}")


def _download_full_and_recent(ticker: str) -> pd.DataFrame:
    _log(f"{ticker}: descàrrega inicial completa ({FULL_DOWNLOAD_PERIOD})")
    df_full = _download_from_yahoo(ticker, period=FULL_DOWNLOAD_PERIOD)

    recent_start = (
        _utc_now_naive().normalize() - pd.Timedelta(days=RECENT_REFRESH_DAYS)
    ).strftime("%Y-%m-%d")

    _log(f"{ticker}: descàrrega recent addicional des de {recent_start}")
    df_recent = _download_from_yahoo(ticker, start=recent_start)

    merged = _merge_frames_by_date(df_full, df_recent)

    _log(
        f"{ticker}: complet={len(df_full)} files, recent={len(df_recent)} files, "
        f"final={len(merged)} files"
    )

    return merged


def _read_local_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return _empty_ohlcv_df()

    df = _ensure_ohlcv_schema(df)
    df = _fill_synthetic_ohlcv(df)
    return df


def _refresh_recent_window(ticker: str, df_local: pd.DataFrame) -> pd.DataFrame:
    """
    Manté intacte l'històric antic del CSV local i només substitueix els últims dies.
    Açò és ideal quan el CSV ve de diverses fonts i és la teua font de veritat.
    """
    df_local = _fill_synthetic_ohlcv(df_local)

    if df_local.empty:
        _log(f"{ticker}: CSV buit, es farà descàrrega completa")
        return _download_full_and_recent(ticker)

    old_last_sig = _last_candle_signature(df_local)

    recent_start_ts = (
        df_local.index.max().normalize() - pd.Timedelta(days=RECENT_REFRESH_DAYS)
    ).normalize()
    recent_start = recent_start_ts.strftime("%Y-%m-%d")

    older_local = df_local.loc[df_local.index < recent_start_ts].copy()
    recent_local = df_local.loc[df_local.index >= recent_start_ts].copy()

    df_recent = _download_from_yahoo(ticker, start=recent_start)

    if df_recent.empty:
        _log(
            f"{ticker}: refresc recent des de {recent_start} sense dades noves. "
            f"Es manté el CSV local."
        )
        return df_local

    # Important: no perdem recents locals si Yahoo no els torna.
    # Les files descarregades al final sobreescriuen els dies solapats.
    merged = _merge_frames_by_date(older_local, recent_local, df_recent)
    new_last_sig = _last_candle_signature(merged)

    _log(
        f"{ticker}: refrescats últims {RECENT_REFRESH_DAYS} dies "
        f"(des de {recent_start}); "
        f"local_recent={len(recent_local)} files, "
        f"downloaded_recent={len(df_recent)} files, "
        f"final={len(merged)} files"
    )

    if old_last_sig != new_last_sig:
        _log(f"{ticker}: última vela antiga={old_last_sig} -> nova={new_last_sig}")

    return merged


def _save_local_csv(path: str, df: pd.DataFrame) -> None:
    out = _fill_synthetic_ohlcv(df)
    out.index.name = "Date"

    tmp_path = f"{path}.tmp"
    out.to_csv(tmp_path)
    os.replace(tmp_path, path)


def _empty_rate_series() -> pd.Series:
    return pd.Series(dtype="float64", name="Rate", index=pd.DatetimeIndex([], name="Date"))


def _read_rate_csv(path: str) -> pd.Series:
    try:
        df = pd.read_csv(path)
    except Exception:
        return _empty_rate_series()

    if df.empty:
        return _empty_rate_series()

    cols = {str(col).strip().lower(): col for col in df.columns}
    date_col = cols.get("date") or df.columns[0]
    rate_col = cols.get("rate")
    if rate_col is None:
        value_cols = [col for col in df.columns if col != date_col]
        if not value_cols:
            return _empty_rate_series()
        rate_col = value_cols[0]

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "Rate": pd.to_numeric(df[rate_col].replace(".", pd.NA), errors="coerce"),
        }
    ).dropna(subset=["Date"])
    out = out.sort_values("Date")
    out["Date"] = out["Date"].dt.normalize()
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.set_index("Date")
    out.index.name = "Date"
    return out["Rate"].astype(float)


def _download_ecbdfr_from_fred() -> pd.Series:
    _log("ECBDFR: descarregant dades de FRED")
    try:
        return _download_ecbdfr_csv()
    except Exception as exc:
        _log(f"ECBDFR: descàrrega CSV de FRED fallida ({exc}). Provant API ECB.")
    try:
        return _download_ecbdfr_ecb_api()
    except Exception as exc:
        _log(f"ECBDFR: descàrrega API ECB fallida ({exc}). Provant taula de text FRED.")
        return _download_ecbdfr_text()


def _download_ecbdfr_csv() -> pd.Series:
    request = urllib.request.Request(
        FRED_ECBDFR_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; borsa-dashboard/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=FRED_TIMEOUT_SECONDS) as response:
        df = pd.read_csv(response)

    if df.empty:
        return _empty_rate_series()

    value_col = "ECBDFR" if "ECBDFR" in df.columns else df.columns[-1]
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["observation_date"], errors="coerce"),
            "Rate": pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce"),
        }
    ).dropna(subset=["Date"])
    out = out.sort_values("Date")
    out["Date"] = out["Date"].dt.normalize()
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.dropna(subset=["Rate"])
    out = out.set_index("Date")
    out.index.name = "Date"
    return out["Rate"].astype(float)


def _download_fred_rate_csv(series_id: str) -> pd.Series:
    series_id = str(series_id).strip().upper()
    url = FRED_SERIES_URLS.get(series_id)
    if not url:
        url = os.environ.get(
            f"FRED_{series_id}_URL",
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        )

    _log(f"{series_id}: descarregant dades de FRED")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; borsa-dashboard/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=FRED_TIMEOUT_SECONDS) as response:
        df = pd.read_csv(response)

    if df.empty:
        return _empty_rate_series()

    cols = {str(col).strip().lower(): col for col in df.columns}
    date_col = cols.get("observation_date") or cols.get("date") or df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[-1]
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "Rate": pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce"),
        }
    ).dropna(subset=["Date"])
    out = out.sort_values("Date")
    out["Date"] = out["Date"].dt.normalize()
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.dropna(subset=["Rate"])
    out = out.set_index("Date")
    out.index.name = "Date"
    return out["Rate"].astype(float)


def _download_federal_reserve_h15_dff() -> pd.Series:
    _log("DFF: descarregant dades alternatives de Federal Reserve H.15")
    request = urllib.request.Request(
        FEDERAL_RESERVE_DFF_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; borsa-dashboard/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=FRED_TIMEOUT_SECONDS) as response:
        df = pd.read_csv(response, skiprows=5)

    if df.empty:
        return _empty_rate_series()

    date_col = df.columns[0]
    value_col = df.columns[-1]
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "Rate": pd.to_numeric(df[value_col].replace({"ND": pd.NA, ".": pd.NA}), errors="coerce"),
        }
    ).dropna(subset=["Date"])
    out = out.sort_values("Date")
    out["Date"] = out["Date"].dt.normalize()
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.dropna(subset=["Rate"])
    out = out.set_index("Date")
    out.index.name = "Date"
    return out["Rate"].astype(float)


def _download_ecbdfr_text() -> pd.Series:
    request = urllib.request.Request(
        FRED_ECBDFR_TEXT_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; borsa-dashboard/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=FRED_TIMEOUT_SECONDS) as response:
        text = response.read().decode("utf-8", errors="replace")

    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(-?\d+(?:\.\d+)?)$", line)
        if match:
            rows.append((match.group(1), float(match.group(2))))

    if not rows:
        return _empty_rate_series()

    out = pd.DataFrame(rows, columns=["Date", "Rate"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).drop_duplicates(subset=["Date"], keep="last")
    out = out.set_index("Date").sort_index()
    out.index.name = "Date"
    return out["Rate"].astype(float)


def _download_ecbdfr_ecb_api() -> pd.Series:
    request = urllib.request.Request(
        ECB_DFR_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; borsa-dashboard/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=FRED_TIMEOUT_SECONDS) as response:
        df = pd.read_csv(response)

    if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return _empty_rate_series()

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["TIME_PERIOD"], errors="coerce"),
            "Rate": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
        }
    ).dropna(subset=["Date", "Rate"])
    out = out.sort_values("Date")
    out["Date"] = out["Date"].dt.normalize()
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.set_index("Date")
    out.index.name = "Date"
    return out["Rate"].astype(float)


def _save_rate_csv(path: str, series: pd.Series) -> None:
    out = series.dropna().copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    df = out.rename("Rate").reset_index()
    df.columns = ["Date", "Rate"]
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _get_fred_rate_series(series_id: str, force_refresh: bool = False, *, allow_stale: bool = False) -> pd.Series:
    series_id = str(series_id).strip().upper()
    path = _rate_csv_path(series_id)
    local = _read_rate_csv(path) if os.path.exists(path) else _empty_rate_series()
    file_ts = _file_timestamp_utc(path)

    needs_refresh = force_refresh or local.empty or ((not allow_stale) and not _is_timestamp_fresh(file_ts))
    if needs_refresh:
        try:
            rate = _download_fred_rate_csv(series_id)
            if rate.empty:
                raise ValueError(f"FRED no ha tornat dades {series_id}.")
            _save_rate_csv(path, rate)
            return rate.copy()
        except Exception as exc:
            if series_id == "DFF":
                try:
                    rate = _download_federal_reserve_h15_dff()
                    if rate.empty:
                        raise ValueError("Federal Reserve H.15 no ha tornat dades DFF.")
                    _save_rate_csv(path, rate)
                    return rate.copy()
                except Exception as fed_exc:
                    _log(f"DFF: fallback Federal Reserve H.15 fallit ({fed_exc}).")
            if not local.empty:
                _log(f"{series_id}: error refrescant FRED ({exc}). Es manté el CSV local.")
                return local.copy()
            raise

    if allow_stale and not _is_timestamp_fresh(file_ts):
        _log(f"{series_id}: usant CSV local fora de finestra de refresc")
    else:
        _log(f"{series_id}: usant CSV local recent sense refrescar")
    return local.copy()


def _try_get_fred_rate_series(series_id: str, force_refresh: bool = False, *, allow_stale: bool = False) -> pd.Series:
    try:
        return _get_fred_rate_series(series_id, force_refresh=force_refresh, allow_stale=allow_stale)
    except Exception as exc:
        _log(f"{series_id}: no disponible ({exc})")
        return _empty_rate_series()


def _get_federal_reserve_dff_series(force_refresh: bool = False, *, allow_stale: bool = False) -> pd.Series:
    path = _rate_csv_path("DFF")
    local = _read_rate_csv(path) if os.path.exists(path) else _empty_rate_series()
    file_ts = _file_timestamp_utc(path)

    needs_refresh = force_refresh or local.empty or ((not allow_stale) and not _is_timestamp_fresh(file_ts))
    if needs_refresh:
        try:
            rate = _download_federal_reserve_h15_dff()
            if rate.empty:
                raise ValueError("Federal Reserve H.15 no ha tornat dades DFF.")
            _save_rate_csv(path, rate)
            return rate.copy()
        except Exception as exc:
            if not local.empty:
                _log(f"DFF: error refrescant Federal Reserve H.15 ({exc}). Es manté el CSV local.")
                return local.copy()
            raise

    if allow_stale and not _is_timestamp_fresh(file_ts):
        _log("DFF: usant CSV local fora de finestra de refresc")
    else:
        _log("DFF: usant CSV local recent sense refrescar")
    return local.copy()


def _combine_fed_rate_series(
    upper: pd.Series,
    lower: pd.Series,
    dff: pd.Series,
    fedfunds: pd.Series,
) -> pd.Series:
    target = pd.DataFrame({"upper": upper, "lower": lower}).dropna()
    target_mid = ((target["upper"] + target["lower"]) / 2.0).rename("target")

    series_by_name = {
        "target": target_mid,
        "dff": dff.rename("dff"),
        "fedfunds": fedfunds.rename("fedfunds"),
    }
    index = pd.DatetimeIndex([], name="Date")
    for series in series_by_name.values():
        if not series.empty:
            index = index.union(pd.DatetimeIndex(series.index))

    if index.empty:
        return _empty_rate_series()

    index = pd.DatetimeIndex(index).sort_values()
    data = pd.DataFrame(index=index)
    for name, series in series_by_name.items():
        data[name] = series.reindex(index).ffill()

    combined = data["target"].combine_first(data["dff"]).combine_first(data["fedfunds"]).dropna()
    combined.name = "Rate"
    combined.index.name = "Date"
    return combined.astype(float)


def get_ecb_deposit_rate(force_refresh: bool = False, *, allow_stale: bool = False) -> pd.Series:
    path = _rate_csv_path("ECBDFR")
    local = _read_rate_csv(path) if os.path.exists(path) else _empty_rate_series()
    file_ts = _file_timestamp_utc(path)

    needs_refresh = force_refresh or local.empty or ((not allow_stale) and not _is_timestamp_fresh(file_ts))
    if needs_refresh:
        try:
            rate = _download_ecbdfr_from_fred()
            if rate.empty:
                raise ValueError("FRED no ha tornat dades ECBDFR.")
            _save_rate_csv(path, rate)
            return rate.copy()
        except Exception as exc:
            if not local.empty:
                _log(f"ECBDFR: error refrescant FRED ({exc}). Es manté el CSV local.")
                return local.copy()
            raise

    if allow_stale and not _is_timestamp_fresh(file_ts):
        _log("ECBDFR: usant CSV local fora de finestra de refresc")
    else:
        _log("ECBDFR: usant CSV local recent sense refrescar")
    return local.copy()


def get_fed_rate(force_refresh: bool = False, *, allow_stale: bool = False) -> pd.Series:
    combined_path = _rate_csv_path("FED_RATE")
    combined_local = _read_rate_csv(combined_path) if os.path.exists(combined_path) else _empty_rate_series()
    combined_ts = _file_timestamp_utc(combined_path)

    if (not force_refresh) and allow_stale and not combined_local.empty:
        if not _is_timestamp_fresh(combined_ts):
            _log("FED_RATE: usant CSV local fora de finestra de refresc")
        else:
            _log("FED_RATE: usant CSV local recent sense refrescar")
        return combined_local.copy()

    dff = _get_federal_reserve_dff_series(force_refresh=force_refresh, allow_stale=allow_stale)
    upper = _empty_rate_series()
    lower = _empty_rate_series()
    fedfunds = _empty_rate_series()

    if dff.empty:
        upper = _try_get_fred_rate_series("DFEDTARU", force_refresh=force_refresh, allow_stale=allow_stale)
        lower = _try_get_fred_rate_series("DFEDTARL", force_refresh=force_refresh, allow_stale=allow_stale)
        dff = _try_get_fred_rate_series("DFF", force_refresh=force_refresh, allow_stale=allow_stale)
        fedfunds = _try_get_fred_rate_series("FEDFUNDS", force_refresh=force_refresh, allow_stale=allow_stale)

    fed_rate = dff.copy() if not dff.empty and upper.empty and lower.empty and fedfunds.empty else _combine_fed_rate_series(
        upper,
        lower,
        dff,
        fedfunds,
    )
    if fed_rate.empty:
        raise ValueError("No s'han pogut obtindre dades FED de FRED.")

    _save_rate_csv(_rate_csv_path("FED_RATE"), fed_rate)
    return fed_rate.copy()


def get_data(ticker: str, force_refresh: bool = False, *, allow_stale: bool = False) -> pd.DataFrame:
    ticker = _normalize_ticker(ticker)
    now = _utc_now_naive()

    if not ticker:
        raise ValueError("Ticker buit o invàlid.")

    if (not force_refresh) and ticker in _CACHE_MEMORIA:
        cached_df, cached_at = _CACHE_MEMORIA[ticker]
        if _is_timestamp_fresh(cached_at):
            _log(f"{ticker}: retornant dades des de memòria cau")
            return cached_df.copy()

    path = _csv_path(ticker)

    if os.path.exists(path):
        df_local = _read_local_csv(path)
        file_ts = _file_timestamp_utc(path)

        if force_refresh or df_local.empty or ((not allow_stale) and not _is_timestamp_fresh(file_ts)):
            if force_refresh:
                _log(f"{ticker}: force_refresh=True, refrescant dades")
            elif df_local.empty:
                _log(f"{ticker}: CSV buit o corrupte, refrescant dades")
            else:
                _log(f"{ticker}: CSV antic, refrescant dades")

            try:
                df = _refresh_recent_window(ticker, df_local)
                _save_local_csv(path, df)
            except Exception as exc:
                if not df_local.empty:
                    _log(
                        f"{ticker}: error refrescant dades ({exc}). "
                        f"Es manté temporalment el CSV local."
                    )
                    df = df_local
                else:
                    raise
        else:
            if allow_stale and not _is_timestamp_fresh(file_ts):
                _log(f"{ticker}: usant CSV local fora de finestra de refresc")
            else:
                _log(f"{ticker}: usant CSV local recent sense refrescar")
            df = df_local
    else:
        _log(f"{ticker}: no existeix CSV local, fent descàrrega inicial")
        df = _download_full_and_recent(ticker)
        _save_local_csv(path, df)

    if df.empty:
        raise ValueError(f"No s'han pogut obtindre dades per a {ticker}.")

    df = _fill_synthetic_ohlcv(df)

    out = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    out.index.name = "Date"

    _CACHE_MEMORIA[ticker] = (out.copy(), _utc_now_naive())
    return out.copy()
