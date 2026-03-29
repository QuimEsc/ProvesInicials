from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


COLOR_SUPPORT = "rgba(239, 68, 68, 0.16)"    # roig
COLOR_RESISTANCE = "rgba(245, 158, 11, 0.16)"  # groc

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


def get_red_zones(df: pd.DataFrame, max_zones: int | None = None) -> list[dict]:
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns) or len(df) < 40:
        return []

    d = df[["open", "high", "low", "close"]].copy()
    if "volume" in df.columns:
        d["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
    else:
        d["volume"] = 1.0

    d["atr"] = _atr_wilder(d, ATR_PERIOD)
    d["vol_ma"] = d["volume"].rolling(20, min_periods=1).mean()
    d["vol_ratio"] = (d["volume"] / (d["vol_ma"] + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    has_volume_signal = bool(
        d["volume"].nunique(dropna=True) > 20
        and float(d["volume"].std(skipna=True) or 0.0) > 1e-9
    )

    swing_high, swing_low = _find_swings_confirmed(d, SWING_LEFT, SWING_RIGHT)
    sh_arr = swing_high.to_numpy()
    sl_arr = swing_low.to_numpy()
    highs = d["high"].to_numpy()
    lows = d["low"].to_numpy()

    pending: list[dict] = []
    raw_zones: list[RawZone] = []

    warmup = max(ATR_PERIOD + SWING_LEFT + SWING_RIGHT + 10, 25)

    for i in range(len(d)):
        if pending:
            new_pending = []
            for p in pending:
                age = i - p["impulse_idx"]
                if age > MAX_WAIT_BOS:
                    continue

                if p["role"] == "support" and float(d["close"].iloc[i]) < p["zone_low"]:
                    continue
                if p["role"] == "resistance" and float(d["close"].iloc[i]) > p["zone_high"]:
                    continue

                broke = float(d["close"].iloc[i]) > p["bos_level"] if p["role"] == "support" else float(d["close"].iloc[i]) < p["bos_level"]
                if broke:
                    score = _compute_score(
                        zone_low=p["zone_low"],
                        zone_high=p["zone_high"],
                        atr_val=float(d["atr"].iloc[i]),
                        fvg_pct=float(p["fvg_pct"]),
                        disp_score=float(p["disp_score"]),
                        base_candles=int(p["base_candles"]),
                        impulse_vol_ratio=float(p["impulse_vol_ratio"]),
                        base_vol_ratio_mean=float(p["base_vol_ratio_mean"]),
                        has_volume_signal=has_volume_signal,
                    )
                    raw_zones.append(
                        RawZone(
                            role=p["role"],
                            start_idx=i,
                            low=float(p["zone_low"]),
                            high=float(p["zone_high"]),
                            center=0.5 * (float(p["zone_low"]) + float(p["zone_high"])),
                            score=score,
                            origin_idx=int(p["base_start_idx"]),
                            confirm_idx=i,
                        )
                    )
                else:
                    new_pending.append(p)
            pending = new_pending

        if i < warmup:
            continue

        o = float(d["open"].iloc[i])
        h = float(d["high"].iloc[i])
        l = float(d["low"].iloc[i])
        c = float(d["close"].iloc[i])
        atr_i = float(d["atr"].iloc[i])

        disp_ok, disp_score = _is_displacement(o, h, l, c, atr_i)
        if not disp_ok:
            continue

        bullish_impulse = c > o
        role = "support" if bullish_impulse else "resistance"

        if has_volume_signal:
            impulse_vol_ratio = float(d["vol_ratio"].iloc[i]) if np.isfinite(d["vol_ratio"].iloc[i]) else 1.0
            if impulse_vol_ratio < MIN_IMPULSE_VOL_RATIO:
                continue
        else:
            impulse_vol_ratio = 1.0

        fvg_pct = _compute_fvg_pct(d, i - 2, role)
        if REQUIRE_FVG and fvg_pct < MIN_FVG_PCT:
            continue

        if role == "support":
            prev = _last_swing_before(i, sh_arr, highs, STRUCTURE_LOOKBACK)
        else:
            prev = _last_swing_before(i, sl_arr, lows, STRUCTURE_LOOKBACK)
        if prev is None:
            continue
        _, bos_level = prev

        base = _find_base_zone(d, i, has_volume_signal=has_volume_signal)
        if base is None:
            continue

        base_start_idx, zone_low, zone_high, base_candles, base_vol_ratio_mean = base
        if (zone_high - zone_low) > MAX_ZONE_ATR_MULT * atr_i:
            continue

        pending.append(
            {
                "role": role,
                "impulse_idx": i,
                "base_start_idx": base_start_idx,
                "zone_low": float(zone_low),
                "zone_high": float(zone_high),
                "base_candles": int(base_candles),
                "bos_level": float(bos_level),
                "fvg_pct": float(fvg_pct),
                "disp_score": float(disp_score),
                "impulse_vol_ratio": float(impulse_vol_ratio),
                "base_vol_ratio_mean": float(base_vol_ratio_mean),
            }
        )

    raw_zones = _merge_raw_zones(raw_zones)

    for zone in raw_zones:
        _simulate_lifecycle(zone, d)

    current_close = float(d["close"].iloc[-1])
    out: list[dict] = []

    for zone in raw_zones:
        end_idx = zone.end_idx if zone.end_idx is not None else (len(d) - 1)
        if end_idx <= zone.start_idx:
            continue

        out.append(
            {
                "start": pd.to_datetime(d.index[zone.start_idx]).strftime("%Y-%m-%d"),
                "end": pd.to_datetime(d.index[end_idx]).strftime("%Y-%m-%d"),
                "low": float(zone.low),
                "high": float(zone.high),
                "center": float(zone.center),
                "role": zone.role,
                "active_now": zone.end_idx is None,
                "touch_count": int(zone.touch_count),
                "score": float(zone.score),
                "end_reason": zone.end_reason,
                "dist_pct": abs(float(zone.center) - current_close) / max(1e-12, current_close),
                "color": COLOR_SUPPORT if zone.role == "support" else COLOR_RESISTANCE,
            }
        )

    out.sort(key=lambda z: (z["start"], z["center"]))

    if max_zones is not None:
        ranked = sorted(
            out,
            key=lambda z: (
                not z["active_now"],
                z["dist_pct"],
                -z["score"],
                z["start"],
            ),
        )
        ranked = ranked[:max_zones]
        out = sorted(ranked, key=lambda z: (z["start"], z["center"]))

    return out