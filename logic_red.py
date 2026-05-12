from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


COLOR_SUPPORT = "rgba(0, 218, 60, 0.18)"
COLOR_RESISTANCE = "rgba(255, 59, 48, 0.18)"

SWING_LEFT = 2
SWING_RIGHT = 2
STRUCTURE_LOOKBACK = 220

ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.6
DISPLACEMENT_BODY_RATIO = 0.60

BASE_LOOKBACK = 12
BASE_MAX_CANDLES = 3
BASE_SMALL_VS_ATR = 0.90

REQUIRE_FVG = True
MIN_FVG_PCT = 0.0015
MAX_WAIT_BOS = 80
MAX_ZONE_ATR_MULT = 2.2

MIN_IMPULSE_VOL_RATIO = 1.15
MAX_BASE_VOL_RATIO = 1.10

BREAK_ATR_MULT = 0.75
REACTION_ATR_MULT = 0.35
MERGE_EPS_PCT = 0.04
MERGE_START_GAP_BARS = 60

PIVOT_WINDOWS = (5, 10, 20, 40, 80)
WEEKLY_PIVOT_WINDOWS = (3, 6, 12, 26, 52)
MONTHLY_PIVOT_WINDOWS = (3, 6, 12, 24)
QUARTERLY_PIVOT_WINDOWS = (2, 4, 8)
RVOL_MA = 20
RVOL_QUANTILE_FILTER = 0.60
MIN_PIVOTS_KEEP = 250
EPS_PCT_MIN = 0.008
EPS_PCT_MAX = 0.050
EPS_ATR_MULT = 2.0
MIN_CLUSTER_PIVOTS = 5
MIN_ZONE_WIDTH_PCT = 0.010
MAX_ZONE_WIDTH_PCT = 0.080
ZONE_QUANTILE_LO = 0.15
ZONE_QUANTILE_HI = 0.85
BOUNCE_HORIZON = 25
TARGET_ATR_MULT = 2.0
CLUSTER_BREAK_ATR_MULT = 1.0
BREAK_PENALTY = 0.25
ENTRY_COOLDOWN_DAYS = 2
MAX_EXPORTED_ZONES = 40
TIMEFRAME_WEIGHTS = {
    "D": 1.0,
    "W": 1.8,
    "M": 2.8,
    "Q": 4.0,
}


@dataclass
class RawZone:
    role: str
    start_idx: int
    low: float
    high: float
    center: float
    score: float
    origin_idx: int
    confirm_idx: int
    end_idx: Optional[int] = None
    end_reason: Optional[str] = None
    touch_count: int = 0


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    rename_map = {}
    for col in out.columns:
        low = str(col).strip().lower()
        if low in {"date", "time"}:
            rename_map[col] = "time"
        elif low in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = low
    if rename_map:
        out = out.rename(columns=rename_map)

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out = out.dropna(subset=["time"]).set_index("time")

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    needed = ["open", "high", "low", "close"]
    if not set(needed).issubset(out.columns):
        return pd.DataFrame(columns=[*needed, "volume"])

    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(1.0)
    else:
        out["volume"] = 1.0

    out = out.dropna(subset=needed).copy()
    out = out[out["close"] > 0]
    return out[[*needed, "volume"]]


def _atr_wilder(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _find_swings_confirmed(df: pd.DataFrame, left: int = SWING_LEFT, right: int = SWING_RIGHT) -> tuple[pd.Series, pd.Series]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(left + right, n):
        k = i - right
        window_high = highs[k - left : k + right + 1]
        window_low = lows[k - left : k + right + 1]

        if len(window_high) == (left + right + 1) and highs[k] == np.max(window_high):
            swing_high[k] = True
        if len(window_low) == (left + right + 1) and lows[k] == np.min(window_low):
            swing_low[k] = True

    return pd.Series(swing_high, index=df.index), pd.Series(swing_low, index=df.index)


def _last_swing_before(idx: int, swing_arr: np.ndarray, levels: np.ndarray, lookback: int) -> Optional[tuple[int, float]]:
    start = max(0, idx - lookback)
    for j in range(idx - 1, start - 1, -1):
        if swing_arr[j]:
            return j, float(levels[j])
    return None


def _is_displacement(o: float, h: float, l: float, c: float, atr_val: float) -> tuple[bool, float]:
    if not np.isfinite(atr_val) or atr_val <= 0:
        return False, 0.0

    rng = h - l
    if rng <= 0:
        return False, 0.0

    body = abs(c - o)
    if rng < DISPLACEMENT_ATR_MULT * atr_val:
        return False, 0.0
    if (body / rng) < DISPLACEMENT_BODY_RATIO:
        return False, 0.0

    score = float((rng / atr_val) * (body / rng))
    return True, score


def _compute_fvg_pct(df: pd.DataFrame, k: int, role: str) -> float:
    if k < 0 or (k + 2) >= len(df):
        return 0.0

    if role == "support":
        gap = float(df["low"].iloc[k + 2] - df["high"].iloc[k])
        ref = float(df["high"].iloc[k])
        return (gap / ref) if (gap > 0 and ref > 0) else 0.0

    gap = float(df["low"].iloc[k] - df["high"].iloc[k + 2])
    ref = float(df["low"].iloc[k])
    return (gap / ref) if (gap > 0 and ref > 0) else 0.0


def _find_base_zone(
    df: pd.DataFrame,
    impulse_idx: int,
    has_volume_signal: bool,
) -> Optional[tuple[int, float, float, int, float]]:
    start = max(0, impulse_idx - BASE_LOOKBACK)
    candidates = list(range(start, impulse_idx))
    if not candidates:
        return None

    base_idxs: list[int] = []
    for k in reversed(candidates):
        atr_k = float(df["atr"].iloc[k])
        if not np.isfinite(atr_k) or atr_k <= 0:
            continue

        vol_ratio_k = float(df["vol_ratio"].iloc[k]) if np.isfinite(df["vol_ratio"].iloc[k]) else 1.0
        if has_volume_signal and vol_ratio_k > MAX_BASE_VOL_RATIO:
            if base_idxs:
                break
            continue

        rng = float(df["high"].iloc[k] - df["low"].iloc[k])
        if rng <= BASE_SMALL_VS_ATR * atr_k:
            base_idxs.append(k)
            if len(base_idxs) >= BASE_MAX_CANDLES:
                break
        else:
            if base_idxs:
                break

    if not base_idxs:
        return None

    base_idxs = sorted(base_idxs)
    zone_low = float(df["low"].iloc[base_idxs].min())
    zone_high = float(df["high"].iloc[base_idxs].max())
    base_vol_mean = float(df["vol_ratio"].iloc[base_idxs].mean()) if has_volume_signal else 1.0
    return base_idxs[0], zone_low, zone_high, len(base_idxs), base_vol_mean


def _compute_score(
    zone_low: float,
    zone_high: float,
    atr_val: float,
    fvg_pct: float,
    disp_score: float,
    base_candles: int,
    impulse_vol_ratio: float,
    base_vol_ratio_mean: float,
    has_volume_signal: bool,
) -> float:
    width = max(1e-12, zone_high - zone_low)
    width_penalty = 1.0 / (1.0 + 2.2 * (width / max(1e-12, atr_val)))
    fvg_component = 1.0 + 60.0 * fvg_pct
    disp_component = 1.0 + min(10.0, disp_score)
    base_component = 1.0 + (0.5 if base_candles == 1 else 0.25 if base_candles == 2 else 0.0)

    if has_volume_signal:
        vol_imp = 1.0 + 0.25 * np.clip((impulse_vol_ratio - 1.0), 0.0, 2.0)
        vol_base = 1.0 + 0.20 * np.clip((1.15 - base_vol_ratio_mean), 0.0, 0.5)
    else:
        vol_imp = 1.0
        vol_base = 1.0

    return float(100.0 * fvg_component * disp_component * base_component * width_penalty * vol_imp * vol_base)


def _merge_raw_zones(zones: list[RawZone]) -> list[RawZone]:
    if not zones:
        return []

    zones = sorted(zones, key=lambda z: (z.role, z.start_idx, z.center))
    merged: list[RawZone] = []

    for zone in zones:
        if not merged:
            merged.append(zone)
            continue

        prev = merged[-1]
        same_role = prev.role == zone.role
        same_epoch = abs(zone.start_idx - prev.start_idx) <= MERGE_START_GAP_BARS
        dist = abs(zone.center - prev.center) / max(1e-12, prev.center)
        overlap = min(prev.high, zone.high) - max(prev.low, zone.low)
        min_width = min(prev.high - prev.low, zone.high - zone.low)
        overlap_ratio = overlap / max(1e-12, min_width)

        if same_role and same_epoch and (dist <= MERGE_EPS_PCT or overlap_ratio >= 0.30):
            prev.low = float(min(prev.low, zone.low))
            prev.high = float(max(prev.high, zone.high))
            prev.center = 0.5 * (prev.low + prev.high)
            prev.start_idx = min(prev.start_idx, zone.start_idx)
            prev.origin_idx = min(prev.origin_idx, zone.origin_idx)
            prev.confirm_idx = min(prev.confirm_idx, zone.confirm_idx)
            prev.score = max(prev.score, zone.score)
        else:
            merged.append(zone)

    return merged


def _simulate_lifecycle(zone: RawZone, df: pd.DataFrame) -> None:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = df["atr"].to_numpy()

    entry_idx = None
    entry_atr = None
    was_inside = False

    for i in range(zone.start_idx, len(df)):
        atr_i = float(atrs[i]) if np.isfinite(atrs[i]) else np.nan
        if not np.isfinite(atr_i) or atr_i <= 0:
            atr_i = max(1e-9, abs(float(closes[i])) * 0.01)

        inside = not (float(highs[i]) < zone.low or float(lows[i]) > zone.high)

        if zone.role == "support":
            if float(closes[i]) < (zone.low - BREAK_ATR_MULT * atr_i):
                zone.end_idx = i
                zone.end_reason = "invalidated"
                return

            if inside and not was_inside and entry_idx is None:
                entry_idx = i
                entry_atr = atr_i
                zone.touch_count += 1
            elif entry_idx is not None and i > entry_idx:
                reaction_buffer = max(REACTION_ATR_MULT * float(entry_atr or atr_i), 0.20 * (zone.high - zone.low))
                reacted = float(closes[i]) > (zone.high + reaction_buffer) or float(lows[i]) > zone.high
                if reacted:
                    zone.end_idx = i
                    zone.end_reason = "used_partial"
                    return

        else:
            if float(closes[i]) > (zone.high + BREAK_ATR_MULT * atr_i):
                zone.end_idx = i
                zone.end_reason = "invalidated"
                return

            if inside and not was_inside and entry_idx is None:
                entry_idx = i
                entry_atr = atr_i
                zone.touch_count += 1
            elif entry_idx is not None and i > entry_idx:
                reaction_buffer = max(REACTION_ATR_MULT * float(entry_atr or atr_i), 0.20 * (zone.high - zone.low))
                reacted = float(closes[i]) < (zone.low - reaction_buffer) or float(highs[i]) < zone.low
                if reacted:
                    zone.end_idx = i
                    zone.end_reason = "used_partial"
                    return

        was_inside = inside


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.nanquantile(values, q))
    cdf = np.cumsum(weights) / total
    return float(np.interp(q, cdf, values))


def _pivot_rows(d: pd.DataFrame, windows: tuple[int, ...], timeframe: str) -> pd.DataFrame:
    rows: list[dict] = []
    tf_weight = TIMEFRAME_WEIGHTS.get(timeframe, 1.0)
    for w in windows:
        if len(d) < (2 * w + 5):
            continue
        highs = d["high"].rolling(window=2 * w + 1, center=True).max()
        lows = d["low"].rolling(window=2 * w + 1, center=True).min()
        high_mask = d["high"].eq(highs)
        low_mask = d["low"].eq(lows)

        for idx, row in d.loc[high_mask].iterrows():
            rows.append({"date": idx, "price": float(row["high"]), "rvol": float(row["rvol"]), "w": w, "timeframe": timeframe, "tf_weight": tf_weight})
        for idx, row in d.loc[low_mask].iterrows():
            rows.append({"date": idx, "price": float(row["low"]), "rvol": float(row["rvol"]), "w": w, "timeframe": timeframe, "tf_weight": tf_weight})

    if not rows:
        return pd.DataFrame()

    piv = pd.DataFrame(rows)
    piv = piv.replace([np.inf, -np.inf], np.nan).dropna(subset=["price"])
    piv = piv[piv["price"] > 0].copy()
    if piv.empty:
        return piv

    piv["weight"] = piv["rvol"].clip(lower=0.1, upper=3.0) * np.sqrt(piv["w"].astype(float)) * piv["tf_weight"].astype(float)
    piv = piv.sort_values("weight", ascending=False)
    keep = max(300, int(len(piv) * (1.0 - RVOL_QUANTILE_FILTER)))
    keep = max(keep, min(MIN_PIVOTS_KEEP, len(piv)))
    return piv.head(keep).reset_index(drop=True)


def _prepare_indicator_frame(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["atr"] = _atr_wilder(out, ATR_PERIOD)
    out["vol_ma"] = out["volume"].rolling(RVOL_MA, min_periods=1).mean()
    out["rvol"] = (out["volume"] / (out["vol_ma"] + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out


def _resample_ohlcv(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = d.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"]).copy()


def _multi_timeframe_pivots(daily: pd.DataFrame) -> pd.DataFrame:
    frames = [
        (_prepare_indicator_frame(daily), PIVOT_WINDOWS, "D"),
        (_prepare_indicator_frame(_resample_ohlcv(daily, "W-FRI")), WEEKLY_PIVOT_WINDOWS, "W"),
        (_prepare_indicator_frame(_resample_ohlcv(daily, "ME")), MONTHLY_PIVOT_WINDOWS, "M"),
        (_prepare_indicator_frame(_resample_ohlcv(daily, "QE")), QUARTERLY_PIVOT_WINDOWS, "Q"),
    ]
    pivots = [_pivot_rows(frame, windows, timeframe) for frame, windows, timeframe in frames if not frame.empty]
    pivots = [p for p in pivots if not p.empty]
    if not pivots:
        return pd.DataFrame()
    return pd.concat(pivots, ignore_index=True)


def _cluster_pivots(d: pd.DataFrame, piv: pd.DataFrame) -> pd.DataFrame:
    if piv.empty:
        return pd.DataFrame()

    atr_pct = (d["atr"] / d["close"]).replace([np.inf, -np.inf], np.nan)
    med_atr_pct = float(np.nanmedian(atr_pct.values))
    if not np.isfinite(med_atr_pct) or med_atr_pct <= 0:
        med_atr_pct = 0.012
    eps = float(np.clip(EPS_ATR_MULT * med_atr_pct, EPS_PCT_MIN, EPS_PCT_MAX))

    piv = piv.sort_values("price").reset_index(drop=True)
    log_prices = np.log(piv["price"].to_numpy())
    groups: list[pd.DataFrame] = []
    start = 0
    for i in range(1, len(piv)):
        if (log_prices[i] - log_prices[i - 1]) > eps:
            groups.append(piv.iloc[start:i])
            start = i
    groups.append(piv.iloc[start:])

    zones: list[dict] = []
    for group in groups:
        if len(group) < MIN_CLUSTER_PIVOTS:
            continue

        prices = group["price"].to_numpy(dtype=float)
        weights = group["weight"].to_numpy(dtype=float)
        center = float(np.exp(np.average(np.log(prices), weights=weights)))
        zlo = _weighted_quantile(prices, weights, ZONE_QUANTILE_LO)
        zhi = _weighted_quantile(prices, weights, ZONE_QUANTILE_HI)
        raw_width = (zhi - zlo) / center if center > 0 else 0.0
        width = float(np.clip(raw_width, MIN_ZONE_WIDTH_PCT, MAX_ZONE_WIDTH_PCT))
        zlo = center * (1.0 - width / 2.0)
        zhi = center * (1.0 + width / 2.0)

        zones.append(
            {
                "center": center,
                "low": float(zlo),
                "high": float(zhi),
                "pivots": int(len(group)),
                "w_pivots": float(np.sum(weights)),
                "tf_count": int(group["timeframe"].nunique()) if "timeframe" in group.columns else 1,
                "timeframes": ",".join(sorted(group["timeframe"].astype(str).unique())) if "timeframe" in group.columns else "D",
                "first_date": pd.to_datetime(group["date"].min()),
                "last_date": pd.to_datetime(group["date"].max()),
            }
        )

    if not zones:
        return pd.DataFrame()

    zdf = pd.DataFrame(zones).sort_values("center").reset_index(drop=True)
    merged: list[dict] = []
    i = 0
    while i < len(zdf):
        row = zdf.iloc[i].to_dict()
        j = i + 1
        while j < len(zdf):
            nxt = zdf.iloc[j].to_dict()
            dist = abs(float(nxt["center"]) - float(row["center"])) / max(1e-12, float(row["center"]))
            width_row = (float(row["high"]) - float(row["low"])) / max(1e-12, float(row["center"]))
            width_nxt = (float(nxt["high"]) - float(nxt["low"])) / max(1e-12, float(nxt["center"]))
            if dist > 0.6 * (width_row + width_nxt):
                break
            if float(nxt["w_pivots"]) > float(row["w_pivots"]):
                row = nxt
            j += 1
        merged.append(row)
        i = j

    return pd.DataFrame(merged)


def _score_cluster_zones(d: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    if zones.empty:
        return zones

    zones = zones.copy().reset_index(drop=True)
    close = d["close"].to_numpy(dtype=float)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    atrv = d["atr"].to_numpy(dtype=float)
    rvol = d["rvol"].to_numpy(dtype=float)

    scored: list[dict] = []
    for _, zone in zones.iterrows():
        zlo = float(zone["low"])
        zhi = float(zone["high"])
        center = float(zone["center"])
        touch_days = 0
        touch_events = 0
        success = 0.0
        fail = 0.0
        break_events = 0
        inside_prev = False
        broken_prev = False
        last_entry = -10_000
        last_touch = pd.NaT

        for i in range(0, max(0, len(d) - BOUNCE_HORIZON - 1)):
            inside = not (high[i] < zlo or low[i] > zhi)
            if inside:
                touch_days += 1
                last_touch = d.index[i]

            if inside and not inside_prev and (i - last_entry >= ENTRY_COOLDOWN_DAYS):
                touch_events += 1
                last_entry = i
                atr_i = float(atrv[i])
                if np.isfinite(atr_i) and atr_i > 0:
                    weight = float(np.clip(rvol[i], 0.3, 3.0))
                    role_at_touch = "support" if close[i] >= center else "resistance"
                    if role_at_touch == "support":
                        target = close[i] + TARGET_ATR_MULT * atr_i
                        break_level = zlo - CLUSTER_BREAK_ATR_MULT * atr_i
                        outcome = 0.0
                        for j in range(i + 1, min(i + BOUNCE_HORIZON + 1, len(d))):
                            if close[j] < break_level:
                                outcome = 0.0
                                break
                            if high[j] >= target:
                                outcome = 1.0
                                break
                    else:
                        target = close[i] - TARGET_ATR_MULT * atr_i
                        break_level = zhi + CLUSTER_BREAK_ATR_MULT * atr_i
                        outcome = 0.0
                        for j in range(i + 1, min(i + BOUNCE_HORIZON + 1, len(d))):
                            if close[j] > break_level:
                                outcome = 0.0
                                break
                            if low[j] <= target:
                                outcome = 1.0
                                break
                    if outcome:
                        success += weight
                    else:
                        fail += weight

            atr_i = float(atrv[i])
            broken = False
            if np.isfinite(atr_i) and atr_i > 0:
                broken = close[i] < (zlo - CLUSTER_BREAK_ATR_MULT * atr_i) or close[i] > (zhi + CLUSTER_BREAK_ATR_MULT * atr_i)
            if broken and not broken_prev:
                break_events += 1

            inside_prev = inside
            broken_prev = broken

        probability = (success + 1.0) / (success + fail + 2.0)
        penalty = 1.0 / (1.0 + BREAK_PENALTY * break_events)
        evidence = 1.0 - np.exp(-0.06 * max(0, touch_days))
        timeframe_boost = 1.0 + 0.18 * max(0, int(zone.get("tf_count", 1)) - 1)
        score = probability * penalty * evidence * (1.0 + 0.10 * np.log1p(float(zone["w_pivots"]))) * timeframe_boost

        item = zone.to_dict()
        item.update(
            {
                "touch_days": int(touch_days),
                "touch_events": int(touch_events),
                "break_events": int(break_events),
                "last_touch": last_touch,
                "probability": float(probability * penalty),
                "score": float(score),
            }
        )
        scored.append(item)

    return pd.DataFrame(scored)


def get_red_zones(df: pd.DataFrame, max_zones: int | None = None) -> list[dict]:
    d = _normalise_ohlcv(df)
    if d.empty or len(d) < 80:
        return []

    d = _prepare_indicator_frame(d)
    piv = _multi_timeframe_pivots(d[["open", "high", "low", "close", "volume"]])
    zones = _cluster_pivots(d, piv)
    zones = _score_cluster_zones(d, zones)
    if zones.empty:
        return []

    current_close = float(d["close"].iloc[-1])
    first_date = pd.to_datetime(d.index[0]).strftime("%Y-%m-%d")
    last_date = pd.to_datetime(d.index[-1]).strftime("%Y-%m-%d")
    zones["dist_pct"] = (zones["center"].astype(float) - current_close).abs() / max(1e-12, current_close)
    zones["role"] = np.where(zones["center"].astype(float) <= current_close, "support", "resistance")
    zones.loc[zones["touch_days"].astype(int) < 5, "score"] *= 0.4

    limit = max_zones if max_zones is not None else MAX_EXPORTED_ZONES
    ranked = zones.sort_values(["dist_pct", "score"], ascending=[True, False]).head(limit)

    out: list[dict] = []
    for _, zone in ranked.sort_values("center").iterrows():
        role = str(zone["role"])
        out.append(
            {
                "start": first_date,
                "end": last_date,
                "low": float(zone["low"]),
                "high": float(zone["high"]),
                "center": float(zone["center"]),
                "role": role,
                "active_now": True,
                "touch_count": int(zone["touch_events"]),
                "touch_days": int(zone["touch_days"]),
                "break_events": int(zone["break_events"]),
                "timeframes": str(zone.get("timeframes", "D")),
                "score": float(zone["score"]),
                "probability": float(zone["probability"]),
                "end_reason": None,
                "dist_pct": float(zone["dist_pct"]),
                "color": COLOR_SUPPORT if role == "support" else COLOR_RESISTANCE,
            }
        )

    return out
