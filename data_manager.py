import os
import re
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

# Logs en consola
ENABLE_LOGS = str(os.environ.get("ENABLE_DATA_MANAGER_LOGS", "1")).strip().lower() not in {"0", "false", "no", "off"}

os.makedirs(CARPETA_DADES, exist_ok=True)

_CACHE_MEMORIA: Dict[str, Tuple[pd.DataFrame, pd.Timestamp]] = {}


def _log(msg: str) -> None:
    if ENABLE_LOGS:
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[data_manager {now}] {msg}")


def _safe_filename(ticker: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(ticker))


def _csv_path(ticker: str) -> str:
    return os.path.join(CARPETA_DADES, f"{_safe_filename(ticker)}.csv")


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
    out = out.sort_index()

    # En dades diàries volem una sola fila per data.
    # Si hi ha dos timestamps del mateix dia, ens quedem amb l'últim.
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].copy()

    out.index.name = "Date"
    return out.sort_index()


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


def _download_from_yahoo(
    ticker: str,
    *,
    start: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    kwargs = {
        "tickers": ticker,
        "interval": YF_INTERVAL,
        "auto_adjust": YF_AUTO_ADJUST,
        "repair": YF_REPAIR,
        "progress": False,
        "threads": False,
    }

    if start is not None:
        kwargs["start"] = start
    else:
        kwargs["period"] = period or FULL_DOWNLOAD_PERIOD

    if yf is None:
        raise ImportError("Falta yfinance. Instal·la yfinance o puja els CSV manuals.")

    df = yf.download(**kwargs)

    if df is None or df.empty:
        return _empty_ohlcv_df()

    df = _ensure_ohlcv_schema(df)
    df = _fill_synthetic_ohlcv(df)
    return df


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

    merged = _merge_frames_by_date(older_local, df_recent)
    new_last_sig = _last_candle_signature(merged)

    last_candle_replaced = old_last_sig != new_last_sig

    _log(
        f"{ticker}: refrescats últims {RECENT_REFRESH_DAYS} dies "
        f"(des de {recent_start}); "
        f"local_recent={len(recent_local)} files, "
        f"downloaded_recent={len(df_recent)} files, "
        f"final={len(merged)} files, "
        f"última vela substituïda={'Sí' if last_candle_replaced else 'No'}"
    )

    if last_candle_replaced:
        _log(f"{ticker}: última vela antiga={old_last_sig} -> nova={new_last_sig}")

    return merged


def _save_local_csv(path: str, df: pd.DataFrame) -> None:
    out = _fill_synthetic_ohlcv(df)
    out.index.name = "Date"
    out.to_csv(path)


def get_data(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    now = _utc_now_naive()

    if (not force_refresh) and ticker in _CACHE_MEMORIA:
        cached_df, cached_at = _CACHE_MEMORIA[ticker]
        if _is_timestamp_fresh(cached_at):
            _log(f"{ticker}: retornant dades des de memòria cau")
            return cached_df.copy()

    path = _csv_path(ticker)

    if os.path.exists(path):
        df_local = _read_local_csv(path)
        file_ts = _file_timestamp_utc(path)

        if force_refresh or not _is_timestamp_fresh(file_ts):
            if force_refresh:
                _log(f"{ticker}: force_refresh=True, refrescant dades")
            else:
                _log(f"{ticker}: CSV antic, refrescant dades")

            df = _refresh_recent_window(ticker, df_local)
            _save_local_csv(path, df)
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

    _CACHE_MEMORIA[ticker] = (out.copy(), now)
    return out.copy()