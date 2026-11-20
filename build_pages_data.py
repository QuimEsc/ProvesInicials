#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from data_manager import REFRESH_INTERVAL_MINUTES, get_data, get_ecb_deposit_rate, get_fed_rate, get_last_fetch_info
import logic_blue
import logic_lines
import logic_ratio
import logic_red
import macro_context
import portfolio_rebalance

ARXIU_TICKERS = "llista_tickers.csv"
DENOMINADOR = "^VIX"
RANG_C = 850
RANG_V = 850
WEEKLY_RULE = "W-FRI"
TIMEFRAME_OPTIONS = ["D", "W"]
REFRESH_TIMEZONE = "Europe/Madrid"
MARKET_REFRESH_START_HOUR = 7
MARKET_REFRESH_END_HOUR = 23
REFRESH_GROUP_MARKET = "market"
REFRESH_GROUP_24H = "24h"
REFRESH_GROUP_DEFAULT = REFRESH_GROUP_MARKET

COLOR_LINIES_COMPRA = {
    "c1": "#93c5fd",
    "c2": "#60a5fa",
    "c3": "#3b82f6",
    "c4": "#1d4ed8",
}
COLOR_LINIES_VENDA = {
    "v1": "#fca5a5",
    "v2": "#f87171",
    "v3": "#ef4444",
    "v4": "#b91c1c",
}


def _assegurar_llista_tickers(base_dir: Path) -> pd.DataFrame:
    path = base_dir / ARXIU_TICKERS
    if not path.exists():
        raise FileNotFoundError(f"No existeix {ARXIU_TICKERS} en {base_dir}")

    df = pd.read_csv(path)
    df = df.dropna(subset=["Nom", "Ticker"]).copy()
    df["Nom"] = df["Nom"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[(df["Nom"] != "") & (df["Ticker"] != "")]
    if df.empty:
        raise ValueError("llista_tickers.csv no conté tickers vàlids.")
    if "RefreshGroup" not in df.columns:
        df["RefreshGroup"] = REFRESH_GROUP_DEFAULT
    df["RefreshGroup"] = df["RefreshGroup"].apply(_normalize_refresh_group)
    return df


def _normalize_refresh_group(value: str | None) -> str:
    group = str(value or "").strip().lower()
    aliases = {
        "": REFRESH_GROUP_DEFAULT,
        "default": REFRESH_GROUP_DEFAULT,
        "mercat": REFRESH_GROUP_MARKET,
        "market_hours": REFRESH_GROUP_MARKET,
        "trading": REFRESH_GROUP_MARKET,
        "always": REFRESH_GROUP_24H,
        "24/7": REFRESH_GROUP_24H,
        "24h": REFRESH_GROUP_24H,
    }
    return aliases.get(group, group if group in {REFRESH_GROUP_MARKET, REFRESH_GROUP_24H} else REFRESH_GROUP_DEFAULT)


def _local_refresh_time(now_utc_naive: pd.Timestamp) -> pd.Timestamp:
    return now_utc_naive.tz_localize("UTC").tz_convert(REFRESH_TIMEZONE)


def _is_refresh_allowed(refresh_group: str, now_local: pd.Timestamp) -> bool:
    group = _normalize_refresh_group(refresh_group)
    if group == REFRESH_GROUP_24H:
        return True
    if now_local.weekday() >= 5:
        return False
    return MARKET_REFRESH_START_HOUR <= int(now_local.hour) < MARKET_REFRESH_END_HOUR


def _parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_localize(None)


def _latest_market_close_utc(now_utc_naive: pd.Timestamp) -> pd.Timestamp:
    now_ny = now_utc_naive.tz_localize("UTC").tz_convert("America/New_York")
    close_today_ny = now_ny.normalize() + pd.Timedelta(hours=16, minutes=20)
    if now_ny.weekday() < 5 and now_ny >= close_today_ny:
        target_ny = close_today_ny
    else:
        cursor = now_ny.normalize() - pd.Timedelta(days=1)
        while cursor.weekday() >= 5:
            cursor -= pd.Timedelta(days=1)
        target_ny = cursor + pd.Timedelta(hours=16, minutes=20)
    return target_ny.tz_convert("UTC").tz_localize(None)


def _needs_post_close_refresh(
    refresh_group: str,
    prev_ticker_entry: dict | None,
    prev_manifest_generated_at: str | None,
    now_utc_naive: pd.Timestamp,
) -> bool:
    if _normalize_refresh_group(refresh_group) == REFRESH_GROUP_24H:
        return False
    last_checked_raw = None
    if isinstance(prev_ticker_entry, dict):
        last_checked_raw = (
            prev_ticker_entry.get("last_checked_at")
            or prev_ticker_entry.get("data_updated_at")
        )
    last_checked = _parse_utc_timestamp(last_checked_raw or prev_manifest_generated_at)
    if last_checked is None:
        return True
    return last_checked < _latest_market_close_utc(now_utc_naive)


def _load_previous_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_git_short_head(base_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=base_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def _slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "ticker"


def _empty_chart_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="str"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
        }
    )


def _to_indexed_ohlc(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out.index = pd.DatetimeIndex([], name="Date")
        return out

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    rename_map = {}
    for col in out.columns:
        low = str(col).strip().lower()
        if low == "open":
            rename_map[col] = "open"
        elif low == "high":
            rename_map[col] = "high"
        elif low == "low":
            rename_map[col] = "low"
        elif low == "close":
            rename_map[col] = "close"
        elif low == "volume":
            rename_map[col] = "volume"
        elif low == "time":
            rename_map[col] = "time"

    if rename_map:
        out = out.rename(columns=rename_map)

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out = out.dropna(subset=["time"]).copy()
        out = out.set_index("time")

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            out[col] = pd.NA

    if "volume" not in out.columns:
        out["volume"] = pd.NA

    out = out[["open", "high", "low", "close", "volume"]].copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_index()
    out.index.name = "Date"
    return out


def _format_candles(df: pd.DataFrame | None) -> pd.DataFrame:
    indexed = _to_indexed_ohlc(df)
    if indexed.empty:
        return _empty_chart_ohlc()

    out = indexed[["open", "high", "low", "close"]].copy().reset_index()
    time_col = out.columns[0]
    out[time_col] = pd.to_datetime(out[time_col]).dt.strftime("%Y-%m-%d")
    out = out.rename(columns={time_col: "time"})
    return out[["time", "open", "high", "low", "close"]]


def _resample_ohlc(df: pd.DataFrame | None, rule: str) -> pd.DataFrame:
    indexed = _to_indexed_ohlc(df)
    if indexed.empty:
        return indexed

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in indexed.columns:
        agg["volume"] = "sum"

    out = indexed.resample(rule).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    out.index.name = "Date"
    return out


def _flat_ratio_from_main(df_main: pd.DataFrame | None) -> pd.DataFrame:
    indexed = _to_indexed_ohlc(df_main)
    if indexed.empty:
        return _empty_chart_ohlc()

    out = pd.DataFrame(index=indexed.index)
    out["open"] = 100.0
    out["high"] = 100.0
    out["low"] = 100.0
    out["close"] = 100.0
    out.index.name = "Date"
    return _format_candles(out)


def _last_close(df: pd.DataFrame | None) -> float | None:
    indexed = _to_indexed_ohlc(df)
    if indexed.empty or "close" not in indexed.columns:
        return None

    s = pd.to_numeric(indexed["close"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _last_line_value(df_line: pd.DataFrame | None, nom_columna: str) -> float | None:
    if df_line is None or df_line.empty:
        return None

    df = df_line.copy()

    if nom_columna not in df.columns:
        cols = [c for c in df.columns if str(c).lower() != "time"]
        if not cols:
            return None
        nom_columna = cols[0]

    s = pd.to_numeric(df[nom_columna], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _next_buy_line(close: float | None, buy_lines: dict[str, float | None]) -> tuple[str | None, float | None, float | None]:
    if close is None:
        return None, None, None

    valides = [(nom, valor) for nom, valor in buy_lines.items() if valor is not None and pd.notna(valor)]
    if not valides:
        return None, None, None

    per_davall = [(nom, valor) for nom, valor in valides if valor <= close]

    if per_davall:
        nom_linia, valor_linia = max(per_davall, key=lambda x: x[1])
    else:
        nom_linia, valor_linia = min(valides, key=lambda x: abs(x[1] - close))

    pct = ((close - valor_linia) / close) * 100 if close else None
    return nom_linia, float(valor_linia), float(pct) if pct is not None else None


def _line_to_generic(df_line: pd.DataFrame | None, line_name: str) -> list[dict]:
    if df_line is None or df_line.empty:
        return []

    df = df_line.copy()
    if line_name not in df.columns:
        value_cols = [c for c in df.columns if str(c).lower() != "time"]
        if not value_cols:
            return []
        line_name = value_cols[0]

    out = df[["time", line_name]].copy()
    out = out.rename(columns={line_name: "value"})
    out["time"] = out["time"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["time", "value"]).copy()
    return out.to_dict(orient="records")


def _clean_for_json(value):
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_for_json(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _clean_for_json(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _clean_for_json(value.tolist())
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean_for_json(payload), ensure_ascii=False, separators=(",", ":"))
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8", newline="")


def _write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8", newline="")


def build_site(base_dir: Path, output_dir: Path, force_refresh: bool = False, *, refresh_rates: bool = False) -> None:
    df_tickers = _assegurar_llista_tickers(base_dir)
    now = pd.Timestamp.utcnow().tz_localize(None)
    generated_at = now.isoformat(timespec="seconds") + "Z"
    now_local = _local_refresh_time(now)
    prev_manifest = _load_previous_manifest(output_dir)
    prev_generated_at = prev_manifest.get("generated_at")
    prev_by_slug = {
        str(item.get("slug")): item
        for item in (prev_manifest.get("tickers") or [])
        if isinstance(item, dict) and item.get("slug")
    }
    prev_by_ticker = {
        str(item.get("ticker")).strip().upper(): item
        for item in (prev_manifest.get("tickers") or [])
        if isinstance(item, dict) and item.get("ticker")
    }
    refresh_groups_by_ticker = {
        str(row["Ticker"]).strip().upper(): _normalize_refresh_group(row.get("RefreshGroup"))
        for row in df_tickers.to_dict(orient="records")
    }
    denominator_refresh_group = refresh_groups_by_ticker.get(DENOMINADOR.upper(), REFRESH_GROUP_DEFAULT)
    denominator_window_open = _is_refresh_allowed(denominator_refresh_group, now_local)
    denominator_post_close = _needs_post_close_refresh(
        denominator_refresh_group,
        prev_by_ticker.get(DENOMINADOR.upper()),
        prev_generated_at,
        now,
    )
    denominator_refresh_allowed = denominator_window_open or denominator_post_close
    df_denominator_daily = None

    output_dir.mkdir(parents=True, exist_ok=True)
    tickers_dir = output_dir / "tickers"
    tickers_dir.mkdir(parents=True, exist_ok=True)

    ticker_items: list[dict] = []
    summary_rows: list[dict] = []
    generated_ticker_files: set[str] = set()
    price_data_by_ticker: dict[str, pd.DataFrame] = {}
    any_post_close_refresh = False
    status_counts = {"updated": 0, "unchanged": 0, "cached": 0, "error": 0}

    for row in df_tickers.to_dict(orient="records"):
        nom = str(row["Nom"]).strip()
        ticker = str(row["Ticker"]).strip()
        refresh_group = _normalize_refresh_group(row.get("RefreshGroup"))
        slug = _slugify(nom)
        prev_entry = prev_by_slug.get(slug) or prev_by_ticker.get(ticker.upper())
        window_open = _is_refresh_allowed(refresh_group, now_local)
        post_close_due = _needs_post_close_refresh(refresh_group, prev_entry, prev_generated_at, now)
        if post_close_due:
            any_post_close_refresh = True
        refresh_allowed = window_open or post_close_due
        data_kwargs = {
            "force_refresh": force_refresh and refresh_allowed,
            "allow_stale": not refresh_allowed,
        }

        print(
            f"Generant {nom} ({ticker})... "
            f"refresh_group={refresh_group}, refresh_allowed={refresh_allowed}, post_close={post_close_due}"
        )

        df_main_daily = get_data(ticker, **data_kwargs)
        fetch_info = get_last_fetch_info(ticker)
        price_data_by_ticker[ticker.upper()] = df_main_daily
        df_main_weekly = _resample_ohlc(df_main_daily, WEEKLY_RULE)

        candles_daily = _format_candles(df_main_daily)
        candles_weekly = _format_candles(df_main_weekly)

        lines_raw = logic_lines.get_lines(df_main_daily, rang_c=RANG_C, rang_v=RANG_V) or {}
        blue_zones_daily = logic_blue.get_blue_zones(df_main_daily, max_zones=None)
        red_zones_daily = logic_red.get_red_zones(df_main_daily, max_zones=None)

        if ticker != DENOMINADOR:
            try:
                if df_denominator_daily is None:
                    df_denominator_daily = get_data(
                        DENOMINADOR,
                        force_refresh=force_refresh and denominator_refresh_allowed,
                        allow_stale=not denominator_refresh_allowed,
                    )
                ratio_daily_raw = logic_ratio.calculate_ratio_data(df_main_daily, df_denominator_daily)
                ratio_daily_indexed = _to_indexed_ohlc(ratio_daily_raw)
                ratio_weekly_indexed = _resample_ohlc(ratio_daily_indexed, WEEKLY_RULE)
                ratio_daily = _format_candles(ratio_daily_indexed)
                ratio_weekly = _format_candles(ratio_weekly_indexed)
                if ratio_daily.empty:
                    ratio_daily = _flat_ratio_from_main(df_main_daily)
                if ratio_weekly.empty:
                    ratio_weekly = _flat_ratio_from_main(df_main_weekly)
            except Exception as exc:
                print(f"  -> ratio fallback per a {ticker}: {exc}")
                ratio_daily = _flat_ratio_from_main(df_main_daily)
                ratio_weekly = _flat_ratio_from_main(df_main_weekly)
        else:
            ratio_daily = _flat_ratio_from_main(df_main_daily)
            ratio_weekly = _flat_ratio_from_main(df_main_weekly)

        close = _last_close(df_main_daily)
        quote_date = str(candles_daily.iloc[-1]["time"]) if not candles_daily.empty else None
        prev_summary = prev_entry.get("summary") if isinstance(prev_entry, dict) and isinstance(prev_entry.get("summary"), dict) else {}
        prev_quote_date = prev_entry.get("quote_date") if isinstance(prev_entry, dict) else None
        prev_close = prev_summary.get("close")
        prev_close_matches = (
            prev_close is not None
            and close is not None
            and round(float(prev_close), 6) == round(float(close), 6)
        )
        price_changed = (
            bool(fetch_info.get("candle_changed"))
            or (prev_entry is None)
            or (prev_quote_date != quote_date)
            or (not prev_close_matches)
        )
        prev_data_updated_at = (
            (prev_entry.get("data_updated_at") if isinstance(prev_entry, dict) else None)
            or prev_generated_at
        )
        data_updated_at = generated_at if price_changed or not prev_data_updated_at else prev_data_updated_at
        last_checked_at = fetch_info.get("checked_at") or (
            (prev_entry.get("last_checked_at") if isinstance(prev_entry, dict) else None)
            or prev_data_updated_at
        )
        refresh_status = fetch_info.get("status") or ("cached_outside_window" if not refresh_allowed else "unchanged")
        if refresh_status == "updated":
            status_counts["updated"] += 1
        elif refresh_status == "unchanged":
            status_counts["unchanged"] += 1
        elif refresh_status in {"fallback_error", "error"}:
            status_counts["error"] += 1
        else:
            status_counts["cached"] += 1

        buy_lines = {
            "c1": _last_line_value(lines_raw.get("c1"), "c1"),
            "c2": _last_line_value(lines_raw.get("c2"), "c2"),
            "c3": _last_line_value(lines_raw.get("c3"), "c3"),
            "c4": _last_line_value(lines_raw.get("c4"), "c4"),
        }
        nom_linia, valor_linia, pct = _next_buy_line(close, buy_lines)

        ticker_payload = {
            "meta": {
                "name": nom,
                "ticker": ticker,
                "slug": slug,
                "denominator": DENOMINADOR,
                "refresh_group": refresh_group,
                "refresh_interval_minutes": REFRESH_INTERVAL_MINUTES,
                "quote_date": quote_date,
            },
            "daily": {
                "candles": candles_daily,
                "ratio": ratio_daily,
                "lines": {
                    "c1": _line_to_generic(lines_raw.get("c1"), "c1"),
                    "c2": _line_to_generic(lines_raw.get("c2"), "c2"),
                    "c3": _line_to_generic(lines_raw.get("c3"), "c3"),
                    "c4": _line_to_generic(lines_raw.get("c4"), "c4"),
                    "v1": _line_to_generic(lines_raw.get("v1"), "v1"),
                    "v2": _line_to_generic(lines_raw.get("v2"), "v2"),
                    "v3": _line_to_generic(lines_raw.get("v3"), "v3"),
                    "v4": _line_to_generic(lines_raw.get("v4"), "v4"),
                },
                "zones": {
                    "blue": blue_zones_daily,
                    "red": red_zones_daily,
                },
            },
            "weekly": {
                "candles": candles_weekly,
                "ratio": ratio_weekly,
            },
            "summary": {
                "close": close,
                "line": nom_linia,
                "buy": valor_linia,
                "pct": pct,
                "quote_date": quote_date,
            },
        }

        ticker_filename = f"{slug}.json"
        generated_ticker_files.add(ticker_filename)
        _write_json(tickers_dir / ticker_filename, ticker_payload)

        ticker_items.append(
            {
                "name": nom,
                "ticker": ticker,
                "slug": slug,
                "refresh_group": refresh_group,
                "quote_date": quote_date,
                "data_updated_at": data_updated_at,
                "last_checked_at": last_checked_at,
                "refresh_status": refresh_status,
                "summary": {
                    "close": close,
                    "line": nom_linia,
                    "buy": valor_linia,
                    "pct": pct,
                },
            }
        )
        summary_rows.append(
            {
                "name": nom,
                "ticker": ticker,
                "slug": slug,
                "close": close,
                "line": nom_linia,
                "buy": valor_linia,
                "pct": pct,
                "quote_date": quote_date,
                "data_updated_at": data_updated_at,
                "last_checked_at": last_checked_at,
                "refresh_status": refresh_status,
            }
        )

    summary_rows.sort(
        key=lambda r: (
            float("inf") if r["pct"] is None or pd.isna(r["pct"]) else float(r["pct"]),
            str(r["name"]),
        )
    )

    prev_rates_ts = _parse_utc_timestamp((prev_manifest.get("diagnostics") or {}).get("rates_updated_at"))
    rates_due = refresh_rates or prev_rates_ts is None or ((now - prev_rates_ts) >= pd.Timedelta(hours=18))
    rates_updated_at = generated_at if rates_due else (prev_manifest.get("diagnostics") or {}).get("rates_updated_at")
    diagnostics = {
        "executed_at": os.environ.get("WORKFLOW_EXECUTED_AT") or generated_at,
        "generated_at": generated_at,
        "rates_updated_at": rates_updated_at,
        "source_commit": _safe_git_short_head(base_dir),
        "run_id": os.environ.get("GITHUB_RUN_ID") or None,
        "run_number": os.environ.get("GITHUB_RUN_NUMBER") or None,
        "event_name": os.environ.get("GITHUB_EVENT_NAME") or "local",
        "schedule": os.environ.get("GITHUB_EVENT_SCHEDULE") or None,
        "market_window_open": _is_refresh_allowed(REFRESH_GROUP_MARKET, now_local),
        "post_close_refresh": any_post_close_refresh,
        "updated_count": status_counts["updated"],
        "unchanged_count": status_counts["unchanged"],
        "cached_count": status_counts["cached"],
        "error_count": status_counts["error"],
    }

    manifest = {
        "generated_at": generated_at,
        "diagnostics": diagnostics,
        "denominator": DENOMINADOR,
        "refresh_interval_minutes": REFRESH_INTERVAL_MINUTES,
        "refresh_schedule": {
            "timezone": REFRESH_TIMEZONE,
            "market_days": "monday-friday",
            "market_start_hour": MARKET_REFRESH_START_HOUR,
            "market_end_hour": MARKET_REFRESH_END_HOUR,
            "always_group": REFRESH_GROUP_24H,
        },
        "timeframes": TIMEFRAME_OPTIONS,
        "line_colors": {
            "buy": COLOR_LINIES_COMPRA,
            "sell": COLOR_LINIES_VENDA,
        },
        "tickers": ticker_items,
    }
    summary = {
        "generated_at": generated_at,
        "diagnostics": diagnostics,
        "rows": summary_rows,
    }

    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "summary.json", summary)
    for stale_path in tickers_dir.glob("*.json"):
        if stale_path.name not in generated_ticker_files:
            stale_path.unlink()

    tracking_inputs = None
    try:
        print(f"Generant estratègia de cartera... refresh_rates={rates_due}")
        ecb_rate = get_ecb_deposit_rate(force_refresh=rates_due, allow_stale=not rates_due)
        fed_rate = get_fed_rate(force_refresh=rates_due, allow_stale=not rates_due)
        rebalance_payload, rebalance_history = portfolio_rebalance.build_rebalance_data(
            price_data_by_ticker[portfolio_rebalance.MSCI_WORLD_TICKER.upper()],
            price_data_by_ticker[portfolio_rebalance.NASDAQ_TICKER.upper()],
            ecb_rate,
            fed_rate,
            generated_at=generated_at,
        )
        _write_json(output_dir / "rebalance.json", rebalance_payload)
        _write_text_if_changed(output_dir / "rebalance_history.csv", rebalance_history.to_csv(index=False))
        tracking_inputs = (
            price_data_by_ticker[portfolio_rebalance.MSCI_WORLD_TICKER.upper()],
            price_data_by_ticker[portfolio_rebalance.NASDAQ_TICKER.upper()], ecb_rate, fed_rate,
        )
    except Exception as exc:
        print(
            "AVÍS: no s'ha pogut actualitzar l'estratègia de cartera. "
            f"Es conserven els últims fitxers de rebalanç. Motiu: {exc}"
        )
    if tracking_inputs is not None:
        # Integrity/registry errors must fail the build, not silently publish a
        # changed frozen rule or rewrite a historical observation.
        from prospective_tracking import record_from_inputs
        print("Registre de seguiment:", record_from_inputs(base_dir, *tracking_inputs, generated_at))
    try:
        macro_window_open = _is_refresh_allowed(REFRESH_GROUP_MARKET, now_local)
        macro_post_close = _needs_post_close_refresh(
            REFRESH_GROUP_MARKET,
            None,
            prev_generated_at,
            now,
        )
        macro_refresh_allowed = macro_window_open or macro_post_close
        print(
            "Generant context macro 10Y/DXY... "
            f"refresh_allowed={macro_refresh_allowed}"
        )
        us10y = get_data(
            macro_context.US10Y_TICKER,
            force_refresh=force_refresh and macro_refresh_allowed,
            allow_stale=not macro_refresh_allowed,
        )
        dxy = get_data(
            macro_context.DXY_TICKER,
            force_refresh=force_refresh and macro_refresh_allowed,
            allow_stale=not macro_refresh_allowed,
        )
        macro_payload = macro_context.build_macro_data(
            us10y,
            dxy,
            generated_at=generated_at,
            force_refresh=force_refresh and macro_refresh_allowed,
            allow_stale=not macro_refresh_allowed,
        )
        _write_json(output_dir / "macro.json", macro_payload)
    except Exception as exc:
        print(
            "AVÍS: no s'ha pogut actualitzar el context macro. "
            f"Es conserva l'últim fitxer macro si existeix. Motiu: {exc}"
        )
    print(f"OK. Dades generades en {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera els JSON estàtics per a GitHub Pages.")
    parser.add_argument("--base-dir", default=".", help="Carpeta arrel del projecte")
    parser.add_argument("--output-dir", default="docs/data", help="Carpeta d'eixida dels JSON")
    parser.add_argument("--force-refresh", action="store_true", help="Força el refresc de dades")
    parser.add_argument("--refresh-rates", action="store_true", help="Refresca també les sèries BCE/FED")
    args = parser.parse_args()

    build_site(
        base_dir=Path(args.base_dir).resolve(),
        output_dir=Path(args.base_dir).resolve() / args.output_dir,
        force_refresh=args.force_refresh,
        refresh_rates=args.refresh_rates,
    )
