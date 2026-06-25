from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
US10Y_TICKER = "^TNX"
DXY_TICKER = "DX-Y.NYB"
GERMANY_10Y_SERIES_ID = "IRLTLT01DEM156N"
GERMANY_10Y_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={GERMANY_10Y_SERIES_ID}"
GERMANY_10Y_CACHE = ROOT / "dades_historiques" / f"{GERMANY_10Y_SERIES_ID}.csv"

LOOKBACK_WEEKS = 26
LOOKBACK_DAYS = LOOKBACK_WEEKS * 7

RATE_WATCH_PP = 0.25
RATE_HOSTILE_PP = 0.75
DXY_WATCH_PCT = 0.0
DXY_HOSTILE_PCT = 5.0


def build_macro_data(
    us10y: pd.DataFrame,
    dxy: pd.DataFrame,
    *,
    generated_at: str,
    force_refresh: bool = False,
    allow_stale: bool = True,
) -> dict[str, Any]:
    us10y_close = _close_series(us10y, "us10y")
    dxy_close = _close_series(dxy, "dxy")
    bund10y_close = _load_germany_10y(force_refresh=force_refresh, allow_stale=allow_stale)

    us_item = _rate_item("us10y", "US 10Y", us10y_close)
    bund_item = _rate_item("bund10y", "Bund 10Y", bund10y_close)
    avg_change = _safe_mean([us_item["change_26w_pp"], bund_item["change_26w_pp"]])
    avg_level = _safe_mean([us_item["value"], bund_item["value"]])
    avg_item = _rate_item_from_values("avg10y", "10Y mitjà", avg_level, avg_change)
    dxy_item = _dxy_item(dxy_close)

    rate_state = avg_item["state"]
    dxy_state = dxy_item["state"]
    status = _macro_status(rate_state, dxy_state)
    active_cell = _matrix_cell(rate_state, dxy_state)

    return {
        "meta": {
            "generated_at": generated_at,
            "lookback_weeks": LOOKBACK_WEEKS,
            "sources": [
                {"label": "US 10Y", "ticker": US10Y_TICKER, "source": "Yahoo Finance"},
                {"label": "DXY", "ticker": DXY_TICKER, "source": "Yahoo Finance"},
                {
                    "label": "Bund 10Y",
                    "ticker": GERMANY_10Y_SERIES_ID,
                    "source": "FRED/OECD",
                },
            ],
            "thresholds": {
                "rate_watch_pp": RATE_WATCH_PP,
                "rate_hostile_pp": RATE_HOSTILE_PP,
                "dxy_watch_pct": DXY_WATCH_PCT,
                "dxy_hostile_pct": DXY_HOSTILE_PCT,
            },
        },
        "status": status,
        "items": [us_item, bund_item, avg_item, dxy_item],
        "matrix": {
            "rate_state": rate_state,
            "dxy_state": dxy_state,
            "active_cell": active_cell,
        },
    }


def _close_series(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64", name=name, index=pd.DatetimeIndex([], name="Date"))

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    cols = {str(col).strip().lower(): col for col in out.columns}
    close_col = cols.get("close") or out.columns[-1]
    idx = pd.to_datetime(out.index, errors="coerce")
    s = pd.Series(pd.to_numeric(out[close_col], errors="coerce").to_numpy(), index=idx, name=name)
    s = s[~s.index.isna()].dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "Date"
    return s.astype(float)


def _load_germany_10y(*, force_refresh: bool, allow_stale: bool) -> pd.Series:
    cache_path = GERMANY_10Y_CACHE
    if not force_refresh and cache_path.exists():
        cached = _read_rate_cache(cache_path, "bund10y")
        if not cached.empty:
            return cached

    try:
        df = pd.read_csv(GERMANY_10Y_URL)
        df = df.rename(columns={"observation_date": "Date", GERMANY_10Y_SERIES_ID: "Close"})
        df = df[["Date", "Close"]].dropna()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        return _read_rate_cache(cache_path, "bund10y")
    except Exception:
        if cache_path.exists():
            return _read_rate_cache(cache_path, "bund10y")
        raise


def _read_rate_cache(path: Path, name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    value_col = "Close" if "Close" in df.columns else df.columns[-1]
    out = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(), index=df["Date"], name=name)
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "Date"
    return out.astype(float)


def _rate_item(key: str, label: str, series: pd.Series) -> dict[str, Any]:
    value, previous, date, previous_date = _latest_and_previous(series)
    change = value - previous if _is_number(value) and _is_number(previous) else None
    return _rate_item_from_values(key, label, value, change, date=date, previous_date=previous_date)


def _rate_item_from_values(
    key: str,
    label: str,
    value: float | None,
    change: float | None,
    *,
    date: str | None = None,
    previous_date: str | None = None,
) -> dict[str, Any]:
    state = _rate_state(change)
    return {
        "key": key,
        "label": label,
        "kind": "rate",
        "value": _round(value),
        "value_unit": "%",
        "change_26w_pp": _round(change),
        "change_unit": "pp",
        "state": state,
        "state_label": _state_label(state),
        "date": date,
        "previous_date": previous_date,
        "meter": _meter(change, RATE_HOSTILE_PP),
    }


def _dxy_item(series: pd.Series) -> dict[str, Any]:
    value, previous, date, previous_date = _latest_and_previous(series)
    change_pct = ((value / previous) - 1.0) * 100.0 if _is_number(value) and _is_number(previous) and previous else None
    state = _dxy_state(change_pct)
    return {
        "key": "dxy",
        "label": "DXY",
        "kind": "dxy",
        "value": _round(value),
        "value_unit": "",
        "change_26w_pct": _round(change_pct),
        "change_unit": "%",
        "state": state,
        "state_label": _state_label(state),
        "date": date,
        "previous_date": previous_date,
        "meter": _meter(change_pct, DXY_HOSTILE_PCT),
    }


def _latest_and_previous(series: pd.Series) -> tuple[float | None, float | None, str | None, str | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return None, None, None, None
    latest_date = clean.index.max()
    lookback_date = latest_date - pd.Timedelta(days=LOOKBACK_DAYS)
    previous = clean[clean.index <= lookback_date]
    if previous.empty:
        return float(clean.iloc[-1]), None, latest_date.date().isoformat(), None
    previous_date = previous.index.max()
    return (
        float(clean.loc[latest_date]),
        float(previous.loc[previous_date]),
        latest_date.date().isoformat(),
        previous_date.date().isoformat(),
    )


def _macro_status(rate_state: str, dxy_state: str) -> dict[str, str]:
    rate_hot = rate_state == "hostile"
    dxy_hot = dxy_state == "hostile"
    if rate_hot and dxy_hot:
        return {
            "key": "hostile",
            "label": "Hostil",
            "tone": "danger",
            "summary": "10Y i dòlar pressionen alhora.",
        }
    if rate_hot:
        return {
            "key": "rates",
            "label": "Tensió tipus",
            "tone": "danger",
            "summary": "Els tipus llargs pressionen, però el dòlar no confirma.",
        }
    if dxy_hot:
        return {
            "key": "dollar",
            "label": "Tensió dòlar",
            "tone": "danger",
            "summary": "El dòlar puja fort, però els tipus llargs no confirmen.",
        }
    if rate_state == "watch" or dxy_state == "watch":
        return {
            "key": "watch",
            "label": "Vigilància",
            "tone": "warning",
            "summary": "Hi ha pressió moderada, sense senyal hostil complet.",
        }
    return {
        "key": "normal",
        "label": "Normal",
        "tone": "good",
        "summary": "Sense pressió macro rellevant.",
    }


def _matrix_cell(rate_state: str, dxy_state: str) -> str:
    rate_up = rate_state in {"watch", "hostile"}
    dxy_up = dxy_state in {"watch", "hostile"}
    if rate_up and dxy_up:
        return "hostile"
    if rate_up:
        return "rates"
    if dxy_up:
        return "dollar"
    return "normal"


def _rate_state(change_pp: float | None) -> str:
    if not _is_number(change_pp):
        return "unknown"
    if change_pp > RATE_HOSTILE_PP:
        return "hostile"
    if change_pp > RATE_WATCH_PP:
        return "watch"
    return "calm"


def _dxy_state(change_pct: float | None) -> str:
    if not _is_number(change_pct):
        return "unknown"
    if change_pct > DXY_HOSTILE_PCT:
        return "hostile"
    if change_pct > DXY_WATCH_PCT:
        return "watch"
    return "calm"


def _state_label(state: str) -> str:
    return {
        "calm": "Tranquil",
        "watch": "Vigilància",
        "hostile": "Pressió",
        "unknown": "Sense dades",
    }.get(state, state)


def _meter(value: float | None, hostile_threshold: float) -> float | None:
    if not _is_number(value):
        return None
    scaled = (float(value) / hostile_threshold) * 100.0 if hostile_threshold else 0.0
    return max(0.0, min(100.0, scaled))


def _safe_mean(values: list[float | None]) -> float | None:
    nums = [float(value) for value in values if _is_number(value)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _is_number(value: float | None) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _round(value: float | None) -> float | None:
    if not _is_number(value):
        return None
    return round(float(value), 4)
