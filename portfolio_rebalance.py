from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


MSCI_WORLD_TICKER = "^990100-USD-STRD"
NASDAQ_TICKER = "^NDX"

REGIME_DEFENSIVE = "defensiu"
REGIME_NORMAL = "normal"
REGIME_RISK_ON = "risc-on"
REGIME_SEMI_DEFENSIVE = "semi-defensiu"

STATE_POSITIVE = "positiu"
STATE_NEUTRAL = "neutral"
STATE_NEGATIVE = "negatiu"

CONFIRMATION_WEEKS = 4
REBALANCE_THRESHOLD_PP = 2.0

BLOCK_WEIGHTS = {
    "msci_trend": 35.56,
    "nasdaq_trend": 22.22,
    "msci_momentum": 13.33,
    "msci_volatility": 8.89,
    "bce_fed_rate": 20.0,
}

REGIME_WEIGHTS = {
    REGIME_DEFENSIVE: {"world": 45.0, "nasdaq": 15.0, "cash": 40.0},
    REGIME_NORMAL: {"world": 60.0, "nasdaq": 25.0, "cash": 15.0},
    REGIME_RISK_ON: {"world": 40.0, "nasdaq": 55.0, "cash": 5.0},
    REGIME_SEMI_DEFENSIVE: {"world": 55.0, "nasdaq": 20.0, "cash": 25.0},
}

ASSET_LABELS = {
    "world": "MSCI World",
    "nasdaq": "Nasdaq",
    "cash": "Monetari",
}


def build_rebalance_data(
    msci_world: pd.DataFrame,
    nasdaq: pd.DataFrame,
    ecb_rate: pd.Series,
    fed_rate: pd.Series,
    *,
    generated_at: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = _prepare_features(msci_world, nasdaq, ecb_rate, fed_rate)
    friday_frame = _weekly_review_frame(features, "W-FRI")
    thursday_frame = _weekly_review_frame(features, "W-THU")

    friday_history = _run_model(friday_frame, "weekly_friday")
    thursday_history = _run_model(thursday_frame, "weekly_thursday")

    current_history = _current_history_frame(friday_frame, features)
    current_rows = _run_model(current_history, "current")

    current_source = current_rows[-1] if current_rows else None
    thursday_source = thursday_history[-1] if thursday_history else None
    friday_source = friday_history[-1] if friday_history else None

    rows_by_official_review = {
        "friday": _panel_rows(current_source, thursday_source, friday_source, friday_history),
        "thursday": _panel_rows(current_source, thursday_source, friday_source, thursday_history),
    }

    payload = {
        "meta": {
            "generated_at": generated_at,
            "model": "MSCI World / Nasdaq-100 / monetari",
            "rate_block": "BCE/FED",
            "default_official_review": "friday",
            "confirmation_weeks": CONFIRMATION_WEEKS,
            "rebalance_threshold_pp": REBALANCE_THRESHOLD_PP,
        },
        "official_review_options": [
            {"value": "friday", "label": "Divendres"},
            {"value": "thursday", "label": "Dijous"},
        ],
        "rows": rows_by_official_review["friday"],
        "rows_by_official_review": rows_by_official_review,
        "history_csv": "./data/rebalance_history.csv",
    }

    history_rows = []
    for row in thursday_history:
        history_rows.append(_history_export_row(_with_official_reference(row, friday_history), "weekly_thursday"))
    for row in friday_history:
        history_rows.append(_history_export_row(row, "weekly_friday"))

    history = pd.DataFrame(history_rows)
    if not history.empty:
        history = history.sort_values(["date", "review_type"]).reset_index(drop=True)
    return payload, history


def _prepare_features(
    msci_world: pd.DataFrame,
    nasdaq: pd.DataFrame,
    ecb_rate: pd.Series,
    fed_rate: pd.Series,
) -> pd.DataFrame:
    msci_close = _close_series(msci_world, "msci_close")
    nasdaq_close = _close_series(nasdaq, "nasdaq_close")
    if msci_close.empty or nasdaq_close.empty:
        raise ValueError("No hi ha prou dades per calcular el model de rebalanç.")

    latest_price_date = max(msci_close.index.max(), nasdaq_close.index.max())
    first_price_date = min(msci_close.index.min(), nasdaq_close.index.min())
    calendar = pd.date_range(first_price_date, latest_price_date, freq="D", name="Date")

    msci_ind = pd.DataFrame(index=msci_close.index)
    msci_ind["msci_close"] = msci_close
    msci_ind["msci_ma200"] = msci_close.rolling(200, min_periods=200).mean()
    msci_ind["msci_ma50"] = msci_close.rolling(50, min_periods=50).mean()
    msci_ind["msci_momentum_252"] = (msci_close / msci_close.shift(252)) - 1.0
    msci_returns = msci_close.pct_change()
    msci_ind["msci_volatility_63"] = msci_returns.rolling(63, min_periods=63).std() * math.sqrt(252)
    drawdown, days_since_high = _drawdown_context(msci_close)
    msci_ind["msci_drawdown"] = drawdown
    msci_ind["days_since_high"] = days_since_high

    nasdaq_ind = pd.DataFrame(index=nasdaq_close.index)
    nasdaq_ind["nasdaq_close"] = nasdaq_close
    nasdaq_ind["nasdaq_ma200"] = nasdaq_close.rolling(200, min_periods=200).mean()
    nasdaq_ind["nasdaq_ma50"] = nasdaq_close.rolling(50, min_periods=50).mean()

    ecb = _rate_series(ecb_rate, "ecb_rate")
    fed = _rate_series(fed_rate, "fed_rate")

    out = pd.DataFrame(index=calendar)
    for col in msci_ind.columns:
        out[col] = msci_ind[col].reindex(calendar).ffill()
    for col in nasdaq_ind.columns:
        out[col] = nasdaq_ind[col].reindex(calendar).ffill()
    out["ecb_rate"] = ecb.reindex(calendar).ffill().fillna(0.0)
    out["fed_rate"] = fed.reindex(calendar).ffill().fillna(0.0)
    out["rate_score"] = (out["ecb_rate"] + out["fed_rate"]) / 2.0

    required = [
        "msci_close",
        "msci_ma200",
        "msci_ma50",
        "msci_momentum_252",
        "msci_volatility_63",
        "nasdaq_close",
        "nasdaq_ma200",
        "nasdaq_ma50",
    ]
    return out.dropna(subset=required).copy()


def _close_series(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64", name=name, index=pd.DatetimeIndex([], name="Date"))
    out = df.copy()
    cols = {str(col).strip().lower(): col for col in out.columns}
    close_col = cols.get("close") or cols.get("Close") or out.columns[-1]
    idx = pd.to_datetime(out.index, errors="coerce")
    s = pd.Series(pd.to_numeric(out[close_col], errors="coerce").to_numpy(), index=idx, name=name)
    s = s[~s.index.isna()].dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "Date"
    return s.astype(float)


def _rate_series(series: pd.Series, name: str) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64", name=name, index=pd.DatetimeIndex([], name="Date"))
    out = series.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = pd.to_numeric(out, errors="coerce").dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.name = name
    out.index.name = "Date"
    return out.astype(float)


def _drawdown_context(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    max_value = -np.inf
    max_date = None
    drawdowns = []
    days = []
    for date, value in close.items():
        value = float(value)
        if value >= max_value:
            max_value = value
            max_date = date
        drawdowns.append((value / max_value) - 1.0 if max_value > 0 else np.nan)
        days.append((date - max_date).days if max_date is not None else np.nan)
    return pd.Series(drawdowns, index=close.index), pd.Series(days, index=close.index)


def _weekly_review_frame(features: pd.DataFrame, rule: str) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    latest_date = features.index.max()
    weekly = features.resample(rule).last()
    weekly = weekly.loc[weekly.index <= latest_date].dropna(subset=["msci_close", "nasdaq_close"])
    weekly.index.name = "Date"
    return weekly


def _current_history_frame(friday_frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return friday_frame.copy()
    current_date = features.index.max()
    if friday_frame.empty:
        return features.loc[[current_date]].copy()
    if current_date in friday_frame.index:
        return friday_frame.loc[friday_frame.index <= current_date].copy()
    if current_date > friday_frame.index.max():
        return pd.concat([friday_frame, features.loc[[current_date]]], axis=0)
    return friday_frame.copy()


def _run_model(review_frame: pd.DataFrame, review_type: str) -> list[dict[str, Any]]:
    current_base = REGIME_NORMAL
    pending_regime = None
    pending_count = 0
    crash_armed = False
    previous_weights = REGIME_WEIGHTS[REGIME_NORMAL]
    rows: list[dict[str, Any]] = []

    for date, row in review_frame.iterrows():
        block_states, score = _score_row(row)
        desired_base = _desired_base_regime(score, current_base)

        if desired_base == current_base:
            pending_regime = None
            pending_count = 0
        else:
            if pending_regime != desired_base:
                pending_regime = desired_base
                pending_count = 1
            else:
                pending_count += 1
            if pending_count >= CONFIRMATION_WEEKS:
                current_base = desired_base
                pending_regime = None
                pending_count = 0

        crash_armed = _update_crash_armed(
            crash_armed,
            float(row["msci_drawdown"]),
            int(row["days_since_high"]),
        )
        final_regime = _final_regime(current_base, crash_armed, score, row)
        weights = REGIME_WEIGHTS[final_regime]
        action = _action(weights, previous_weights)

        item = {
            "review_type": review_type,
            "date": _date_str(date),
            "score": float(round(score, 2)),
            "blocks": block_states,
            "desired_base_regime": desired_base,
            "pending_regime": pending_regime,
            "pending_count": int(pending_count),
            "base_regime": current_base,
            "crash_fast_armed": bool(crash_armed),
            "final_regime": final_regime,
            "weights": dict(weights),
            "reference_weights": dict(previous_weights),
            "diffs": action["diffs"],
            "action_required": action["required"],
            "action": action["text"],
            "metrics": _metrics(row),
        }
        rows.append(item)
        previous_weights = weights

    return rows


def _score_row(row: pd.Series) -> tuple[dict[str, dict[str, Any]], float]:
    states = {
        "msci_trend": _trend_state(row["msci_close"], row["msci_ma200"]),
        "nasdaq_trend": _trend_state(row["nasdaq_close"], row["nasdaq_ma200"]),
        "msci_momentum": _momentum_state(row["msci_momentum_252"]),
        "msci_volatility": _volatility_state(row["msci_volatility_63"]),
        "bce_fed_rate": _rate_state(row["rate_score"]),
    }
    blocks = {}
    score = 0.0
    for key, state in states.items():
        contribution = _state_score(state, BLOCK_WEIGHTS[key])
        blocks[key] = {
            "state": state,
            "weight": BLOCK_WEIGHTS[key],
            "contribution": contribution,
        }
        score += contribution
    return blocks, score


def _trend_state(close: float, ma200: float) -> str:
    if close > ma200 * 1.03:
        return STATE_POSITIVE
    if close < ma200 * 0.97:
        return STATE_NEGATIVE
    return STATE_NEUTRAL


def _momentum_state(momentum: float) -> str:
    if momentum > 0.05:
        return STATE_POSITIVE
    if momentum < -0.05:
        return STATE_NEGATIVE
    return STATE_NEUTRAL


def _volatility_state(volatility: float) -> str:
    if volatility <= 0.20:
        return STATE_POSITIVE
    if volatility <= 0.30:
        return STATE_NEUTRAL
    return STATE_NEGATIVE


def _rate_state(rate: float) -> str:
    if rate <= 4.0:
        return STATE_POSITIVE
    if rate <= 6.0:
        return STATE_NEUTRAL
    return STATE_NEGATIVE


def _state_score(state: str, weight: float) -> float:
    if state == STATE_POSITIVE:
        return float(weight)
    if state == STATE_NEUTRAL:
        return float(weight) / 2.0
    return 0.0


def _desired_base_regime(score: float, current_base: str) -> str:
    if score < 40.0:
        return REGIME_DEFENSIVE
    if score > 75.0:
        return REGIME_RISK_ON
    if current_base == REGIME_DEFENSIVE and score <= 50.0:
        return REGIME_DEFENSIVE
    if current_base == REGIME_RISK_ON and score >= 65.0:
        return REGIME_RISK_ON
    return REGIME_NORMAL


def _update_crash_armed(current: bool, drawdown: float, days_since_high: int) -> bool:
    if current and drawdown >= -0.05:
        return False
    if (not current) and drawdown <= -0.20 and days_since_high <= 90:
        return True
    return current


def _final_regime(base_regime: str, crash_armed: bool, score: float, row: pd.Series) -> str:
    final = base_regime
    if crash_armed and base_regime == REGIME_DEFENSIVE:
        final = REGIME_SEMI_DEFENSIVE

    has_fast_reentry = (
        crash_armed
        and score > 50.0
        and (
            float(row["msci_close"]) > float(row["msci_ma50"])
            or float(row["nasdaq_close"]) > float(row["nasdaq_ma50"])
        )
        and REGIME_WEIGHTS[final]["cash"] > 15.0
    )
    if has_fast_reentry:
        return REGIME_NORMAL
    return final


def _action(weights: dict[str, float], reference_weights: dict[str, float]) -> dict[str, Any]:
    diffs = {asset: float(round(weights[asset] - reference_weights[asset], 4)) for asset in REGIME_WEIGHTS[REGIME_NORMAL]}
    max_abs = max(abs(value) for value in diffs.values())
    if max_abs < REBALANCE_THRESHOLD_PP:
        return {"required": False, "diffs": diffs, "text": "No cal rebalancejar"}

    parts = []
    for asset, diff in diffs.items():
        if abs(diff) < 0.01:
            continue
        verb = "Comprar" if diff > 0 else "Vendre"
        parts.append(f"{verb} {abs(diff):.0f} pp {ASSET_LABELS[asset]}")
    return {"required": True, "diffs": diffs, "text": "; ".join(parts)}


def _with_official_reference(row: dict[str, Any] | None, friday_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if row is None:
        return None
    reference = _official_reference(row["date"], friday_history)
    if reference is None:
        return row
    action = _action(row["weights"], reference)
    out = dict(row)
    out["reference_weights"] = dict(reference)
    out["diffs"] = action["diffs"]
    out["action_required"] = action["required"]
    out["action"] = action["text"]
    return out


def _panel_rows(
    current_row: dict[str, Any] | None,
    thursday_row: dict[str, Any] | None,
    friday_row: dict[str, Any] | None,
    official_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _display_row("Actual 30 min", _with_official_reference(current_row, official_history)),
        _display_row("Setmanal dijous close", _with_official_reference(thursday_row, official_history)),
        _display_row("Setmanal divendres close", _with_official_reference(friday_row, official_history)),
    ]


def _official_reference(date: str, friday_history: list[dict[str, Any]]) -> dict[str, float] | None:
    selected = [row for row in friday_history if row["date"] <= date]
    if not selected:
        return dict(REGIME_WEIGHTS[REGIME_NORMAL])
    latest = selected[-1]
    if latest["date"] == date:
        return dict(latest["reference_weights"])
    return dict(latest["weights"])


def _display_row(label: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"label": label, "available": False}
    out = dict(row)
    out["label"] = label
    out["available"] = True
    return out


def _history_export_row(row: dict[str, Any], review_type: str) -> dict[str, Any]:
    weights = row["weights"]
    diffs = row["diffs"]
    blocks = row["blocks"]
    return {
        "review_type": review_type,
        "date": row["date"],
        "score": row["score"],
        "msci_trend": blocks["msci_trend"]["state"],
        "nasdaq_trend": blocks["nasdaq_trend"]["state"],
        "msci_momentum": blocks["msci_momentum"]["state"],
        "msci_volatility": blocks["msci_volatility"]["state"],
        "bce_fed_rate": blocks["bce_fed_rate"]["state"],
        "desired_base_regime": row["desired_base_regime"],
        "pending_regime": row["pending_regime"],
        "pending_count": row["pending_count"],
        "base_regime": row["base_regime"],
        "crash_fast_armed": row["crash_fast_armed"],
        "final_regime": row["final_regime"],
        "world_weight": weights["world"],
        "nasdaq_weight": weights["nasdaq"],
        "cash_weight": weights["cash"],
        "world_diff": diffs["world"],
        "nasdaq_diff": diffs["nasdaq"],
        "cash_diff": diffs["cash"],
        "action_required": row["action_required"],
        "action": row["action"],
    }


def _metrics(row: pd.Series) -> dict[str, float | int | None]:
    keys = [
        "msci_close",
        "msci_ma200",
        "msci_ma50",
        "nasdaq_close",
        "nasdaq_ma200",
        "nasdaq_ma50",
        "msci_momentum_252",
        "msci_volatility_63",
        "ecb_rate",
        "fed_rate",
        "rate_score",
        "msci_drawdown",
        "days_since_high",
    ]
    out = {}
    for key in keys:
        value = row.get(key)
        if pd.isna(value):
            out[key] = None
        elif key == "days_since_high":
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
