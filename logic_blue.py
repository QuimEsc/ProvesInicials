from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


COLOR_BUY = "rgba(59, 130, 246, 0.18)"    # blau
COLOR_SELL = "rgba(139, 92, 246, 0.18)"   # morat

PIVOT_WINDOW = 20
MIN_PIVOTS_TO_ACTIVATE = 4
CANDIDATE_MAX_AGE = 260

MIN_ZONE_WIDTH_PCT = 0.008
MAX_ZONE_WIDTH_PCT = 0.060
EPS_PCT_MIN = 0.008
EPS_PCT_MAX = 0.050

BREAK_ATR_MULT = 0.75
REACTION_ATR_MULT = 0.25


@dataclass
class CandidateCluster:
    role: str
    first_pivot_idx: int
    last_update_idx: int
    prices: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


@dataclass
class LiveBand:
    role: str
    start_idx: int
    low: float
    high: float
    center: float
    score: float
    pivot_count: int
    color: str
    end_idx: Optional[int] = None
    end_reason: Optional[str] = None
    touch_count: int = 0
    entry_idx: Optional[int] = None
    entry_atr: Optional[float] = None
    was_inside: bool = False


def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return np.nan

    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cum_w = np.cumsum(w)
    total = cum_w[-1]
    if total <= 0:
        return float(np.quantile(v, q))
    target = q * total
    return float(np.interp(target, cum_w, v))


def _cluster_bounds(prices: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    center = float(np.average(prices, weights=weights))
    lo = _weighted_quantile(prices, weights, 0.15)
    hi = _weighted_quantile(prices, weights, 0.85)

    if not np.isfinite(center) or center <= 0:
        center = float(np.mean(prices))

    raw_width_pct = (hi - lo) / center if center > 0 else 0.0
    width_pct = float(np.clip(raw_width_pct, MIN_ZONE_WIDTH_PCT, MAX_ZONE_WIDTH_PCT))
    half = width_pct / 2.0

    low = center * (1.0 - half)
    high = center * (1.0 + half)
    return float(low), float(high), float(center)


def _make_band(candidate: CandidateCluster, activation_idx: int) -> LiveBand:
    prices = np.asarray(candidate.prices, dtype=float)
    weights = np.asarray(candidate.weights, dtype=float)
    low, high, center = _cluster_bounds(prices, weights)

    score = float(np.log1p(len(prices)) * max(1.0, float(weights.sum())))
    color = COLOR_BUY if candidate.role == "buy" else COLOR_SELL

    return LiveBand(
        role=candidate.role,
        start_idx=activation_idx,
        low=low,
        high=high,
        center=center,
        score=score,
        pivot_count=len(prices),
        color=color,
    )


def _register_pivot(
    candidates: list[CandidateCluster],
    role: str,
    pivot_idx: int,
    current_idx: int,
    price: float,
    weight: float,
    atr_pct_ref: float,
) -> Optional[LiveBand]:
    eps_pct = float(np.clip(2.0 * atr_pct_ref, EPS_PCT_MIN, EPS_PCT_MAX))

    best_idx = None
    best_dist = None

    for idx, cand in enumerate(candidates):
        if cand.role != role:
            continue

        cand_center = float(np.average(cand.prices, weights=np.clip(cand.weights, 0.1, None)))
        if cand_center <= 0:
            continue

        dist = abs(price - cand_center) / cand_center
        if dist <= eps_pct and (best_dist is None or dist < best_dist):
            best_idx = idx
            best_dist = dist

    if best_idx is None:
        candidates.append(
            CandidateCluster(
                role=role,
                first_pivot_idx=pivot_idx,
                last_update_idx=current_idx,
                prices=[price],
                weights=[weight],
            )
        )
        return None

    cand = candidates[best_idx]
    cand.prices.append(price)
    cand.weights.append(weight)
    cand.last_update_idx = current_idx

    if len(cand.prices) >= MIN_PIVOTS_TO_ACTIVATE:
        band = _make_band(cand, activation_idx=current_idx)
        del candidates[best_idx]
        return band

    return None


def _update_band_lifecycle(
    band: LiveBand,
    i: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
) -> None:
    if band.end_idx is not None or i < band.start_idx:
        return

    atr_i = float(atrs[i]) if np.isfinite(atrs[i]) else np.nan
    if not np.isfinite(atr_i) or atr_i <= 0:
        atr_i = max(1e-9, abs(float(closes[i])) * 0.01)

    low_band = float(band.low)
    high_band = float(band.high)

    inside = not (float(highs[i]) < low_band or float(lows[i]) > high_band)

    if band.role == "buy":
        if float(closes[i]) < (low_band - BREAK_ATR_MULT * atr_i):
            band.end_idx = i
            band.end_reason = "invalidated"
            band.was_inside = inside
            return

        if inside and not band.was_inside and band.entry_idx is None:
            band.entry_idx = i
            band.entry_atr = atr_i
            band.touch_count += 1
        elif band.entry_idx is not None and i > band.entry_idx:
            reaction_buffer = REACTION_ATR_MULT * float(band.entry_atr or atr_i)
            reacted = float(closes[i]) > (high_band + reaction_buffer) or float(lows[i]) > high_band
            if reacted:
                band.end_idx = i
                band.end_reason = "used_partial"
                band.was_inside = inside
                return

    else:
        if float(closes[i]) > (high_band + BREAK_ATR_MULT * atr_i):
            band.end_idx = i
            band.end_reason = "invalidated"
            band.was_inside = inside
            return

        if inside and not band.was_inside and band.entry_idx is None:
            band.entry_idx = i
            band.entry_atr = atr_i
            band.touch_count += 1
        elif band.entry_idx is not None and i > band.entry_idx:
            reaction_buffer = REACTION_ATR_MULT * float(band.entry_atr or atr_i)
            reacted = float(closes[i]) < (low_band - reaction_buffer) or float(highs[i]) < low_band
            if reacted:
                band.end_idx = i
                band.end_reason = "used_partial"
                band.was_inside = inside
                return

    band.was_inside = inside


def get_blue_zones(
    df: pd.DataFrame,
    max_zones: int | None = None,
    pivot_window: int = PIVOT_WINDOW,
) -> list[dict]:
    needed = {"high", "low", "close"}
    if not needed.issubset(df.columns) or len(df) < (2 * pivot_window + 10):
        return []

    d = df[["high", "low", "close"]].copy()
    if "volume" in df.columns:
        d["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
    else:
        d["volume"] = 1.0

    d["atr"] = _atr_wilder(d, period=14)
    d["vol_sma"] = d["volume"].rolling(20, min_periods=1).mean()
    d["rvol"] = (d["volume"] / d["vol_sma"]).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.2, upper=5.0)

    highs = d["high"].to_numpy()
    lows = d["low"].to_numpy()
    closes = d["close"].to_numpy()
    atrs = d["atr"].to_numpy()
    rvols = d["rvol"].to_numpy()
    dates = d.index

    w = int(pivot_window)
    roll_high = d["high"].rolling(window=2 * w + 1, center=True).max()
    roll_low = d["low"].rolling(window=2 * w + 1, center=True).min()

    high_pivot_mask = d["high"].eq(roll_high) & roll_high.notna()
    low_pivot_mask = d["low"].eq(roll_low) & roll_low.notna()

    candidates: list[CandidateCluster] = []
    bands: list[LiveBand] = []

    for i in range(len(d)):
        for band in bands:
            _update_band_lifecycle(band, i, highs, lows, closes, atrs)

        candidates = [c for c in candidates if (i - c.last_update_idx) <= CANDIDATE_MAX_AGE]

        if i < w:
            continue

        k = i - w
        atr_pct_ref = float(atrs[i] / max(1e-12, closes[i])) if np.isfinite(atrs[i]) else 0.015

        if bool(low_pivot_mask.iloc[k]):
            band = _register_pivot(
                candidates=candidates,
                role="buy",
                pivot_idx=k,
                current_idx=i,
                price=float(lows[k]),
                weight=float(rvols[k]) if np.isfinite(rvols[k]) else 1.0,
                atr_pct_ref=atr_pct_ref,
            )
            if band is not None:
                bands.append(band)

        if bool(high_pivot_mask.iloc[k]):
            band = _register_pivot(
                candidates=candidates,
                role="sell",
                pivot_idx=k,
                current_idx=i,
                price=float(highs[k]),
                weight=float(rvols[k]) if np.isfinite(rvols[k]) else 1.0,
                atr_pct_ref=atr_pct_ref,
            )
            if band is not None:
                bands.append(band)

    current_close = float(closes[-1])
    out: list[dict] = []

    for band in bands:
        start_idx = int(band.start_idx)
        end_idx = int(band.end_idx) if band.end_idx is not None else len(d) - 1
        if end_idx <= start_idx:
            continue

        out.append(
            {
                "start": pd.to_datetime(dates[start_idx]).strftime("%Y-%m-%d"),
                "end": pd.to_datetime(dates[end_idx]).strftime("%Y-%m-%d"),
                "low": float(band.low),
                "high": float(band.high),
                "center": float(band.center),
                "role": band.role,
                "active_now": band.end_idx is None,
                "touch_count": int(band.touch_count),
                "score": float(band.score),
                "end_reason": band.end_reason,
                "dist_pct": abs(float(band.center) - current_close) / max(1e-12, current_close),
                "color": band.color,
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