#!/usr/bin/env python3
"""Freshness watchdog for the dashboard data pipeline.

Evaluates data freshness strictly against the *effective source check /
pipeline heartbeat* (never the age of an unchanged market quote) within market
operating hours (`07:00-23:00 Europe/Madrid`, Mon-Fri, plus pending US post-close
capture).

Criteria:
- Target cadence: ~30 min in market hours, tolerating up to 60 min without error.
- Silent preventive self-heal threshold: 45 min (`needs_self_heal = True`).
- Primary freshness breach threshold: > 60 min (`stale = True`, exit code 2).
- Outside market hours (once post-close is captured): always healthy and silent.

Pure standard library; no market data and no model code is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_MAX_AGE_MIN = 60.0
DEFAULT_SELF_HEAL_MIN = 45.0
DEFAULT_PAGES_LAG_MIN = 20.0
DEFAULT_PAGES_URL = "https://quimesc.github.io/ProvesInicials/docs/data/manifest.json"
SOURCE_HEALTH_ASSET = "BTC-USD"  # 24h asset: informational check during active runs
SOURCE_HEALTH_MAX_AGE_MIN = 180.0
REFRESH_TZ = ZoneInfo("Europe/Madrid")
US_MARKET_TZ = ZoneInfo("America/New_York")
MARKET_START_HOUR = 7
MARKET_END_HOUR = 23


def parse_utc(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def age_minutes(value, now):
    ts = value if isinstance(value, datetime) else parse_utc(value)
    if ts is None:
        return None
    return (now - ts).total_seconds() / 60.0


def latest_market_close_utc(now_utc: datetime) -> datetime:
    """Mirror build_pages_data._latest_market_close_utc (16:20 America/New_York)."""
    now_ny = now_utc.astimezone(US_MARKET_TZ)
    close_today_ny = now_ny.replace(hour=16, minute=20, second=0, microsecond=0)
    if now_ny.weekday() < 5 and now_ny >= close_today_ny:
        target_ny = close_today_ny
    else:
        cursor = now_ny.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        while cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
        target_ny = cursor.replace(hour=16, minute=20, second=0, microsecond=0)
    return target_ny.astimezone(timezone.utc)


def _effective_market_check_ts(repo_manifest: dict) -> datetime | None:
    """Return the latest effective source check timestamp for market assets.

    Falls back to ``generated_at`` for manifests/fixtures without per-ticker
    ``last_checked_at`` entries.
    """
    tickers = repo_manifest.get("tickers") or []
    market_checks: list[datetime] = []
    for entry in tickers:
        if not isinstance(entry, dict):
            continue
        group = str(entry.get("refresh_group") or "market").strip().lower()
        if group == "24h" or entry.get("ticker") == SOURCE_HEALTH_ASSET:
            continue
        ts = parse_utc(entry.get("last_checked_at"))
        if ts is not None:
            market_checks.append(ts)
    if market_checks:
        return max(market_checks)
    return parse_utc(repo_manifest.get("generated_at"))


def evaluate(
    repo_manifest,
    pages_manifest,
    now,
    *,
    max_age_min=DEFAULT_MAX_AGE_MIN,
    self_heal_min=DEFAULT_SELF_HEAL_MIN,
    pages_lag_min=DEFAULT_PAGES_LAG_MIN,
    check_pages=True,
):
    """Return a report dict measuring pipeline freshness within market hours."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    repo_gen_ts = parse_utc(repo_manifest.get("generated_at"))
    repo_age = age_minutes(repo_gen_ts, now)
    source_check_ts = _effective_market_check_ts(repo_manifest)
    source_age = age_minutes(source_check_ts, now)

    now_local = now.astimezone(REFRESH_TZ)
    window_open = (now_local.weekday() < 5) and (MARKET_START_HOUR <= now_local.hour < MARKET_END_HOUR)
    latest_close_utc = latest_market_close_utc(now)
    reference_ts = source_check_ts or repo_gen_ts
    post_close_pending = reference_ts is None or reference_ts < latest_close_utc

    diag = repo_manifest.get("diagnostics") or {}
    market_active = window_open or post_close_pending

    # Measure effective elapsed time *within* the active market window so opening
    # at 07:00 Europe/Madrid after an overnight close does not falsely report
    # 8+ hours of staleness.
    effective_age: float | None
    if reference_ts is None:
        effective_age = None
    elif not market_active:
        effective_age = round(source_age if source_age is not None else repo_age, 1)
    elif window_open:
        today_open_utc = now_local.replace(
            hour=MARKET_START_HOUR, minute=0, second=0, microsecond=0
        ).astimezone(timezone.utc)
        if reference_ts >= today_open_utc:
            effective_age = round(max(0.0, (now - reference_ts).total_seconds() / 60.0), 1)
        elif not post_close_pending or reference_ts >= latest_close_utc:
            # Prior session close was already captured; count only time since today's open.
            effective_age = round(max(0.0, (now - today_open_utc).total_seconds() / 60.0), 1)
        else:
            effective_age = round(max(0.0, (now - reference_ts).total_seconds() / 60.0), 1)
    else:
        # Outside 07-23 Europe/Madrid, only time since the pending close counts.
        effective_age = round(max(0.0, (now - latest_close_utc).total_seconds() / 60.0), 1)

    error_count = int(diag.get("error_count") or 0)
    # If the pipeline just ran recently inside market hours but the external
    # source failed for all market tickers, avoid misreporting a scheduler gap
    # when the cause is the upstream data provider.
    recent_run = repo_age is not None and repo_age <= max_age_min
    source_unavailable_recently = (
        market_active
        and recent_run
        and error_count > 0
        and int(diag.get("updated_count") or 0) == 0
        and int(diag.get("unchanged_count") or 0) == 0
    )

    if not market_active:
        needs_self_heal = False
        stale = False
    elif source_unavailable_recently:
        needs_self_heal = False
        stale = False
    elif effective_age is None:
        needs_self_heal = True
        stale = True
    else:
        needs_self_heal = effective_age > self_heal_min
        stale = effective_age > max_age_min

    report = {
        "now": now.isoformat(timespec="seconds"),
        "market_active": market_active,
        "window_open": window_open,
        "post_close_pending": post_close_pending,
        "max_age_min": max_age_min,
        "self_heal_min": self_heal_min,
        "repo_generated_at": repo_manifest.get("generated_at"),
        "repo_age_min": None if repo_age is None else round(repo_age, 1),
        "effective_checked_at": reference_ts.isoformat().replace("+00:00", "Z") if reference_ts else None,
        "effective_age_min": effective_age,
        "needs_self_heal": needs_self_heal,
        "stale": stale,
        "warnings": [],
        "last_run": {
            "event_name": diag.get("event_name"),
            "run_number": diag.get("run_number"),
            "executed_at": diag.get("executed_at"),
        },
    }

    if error_count > 0:
        report["warnings"].append(
            f"Font financera amb {error_count} error(s) en l'últim refresc; es conserven les últimes dades vàlides."
        )

    if not check_pages:
        report["pages_generated_at"] = None
    elif pages_manifest is None:
        report["pages_generated_at"] = None
        if market_active:
            report["warnings"].append("No s'ha pogut llegir el manifest publicat a GitHub Pages.")
    else:
        pages_gen = pages_manifest.get("generated_at")
        report["pages_generated_at"] = pages_gen
        pages_ts = parse_utc(pages_gen)
        if repo_gen_ts and pages_ts and repo_gen_ts > pages_ts:
            lag = (repo_gen_ts - pages_ts).total_seconds() / 60.0
            if repo_age is not None and repo_age > pages_lag_min:
                report["pages_stale"] = True
                report["warnings"].append(
                    f"GitHub Pages serveix dades {lag:.0f} min més antigues que el repositori."
                )

    entry = next((t for t in repo_manifest.get("tickers", []) if t.get("ticker") == SOURCE_HEALTH_ASSET), None)
    if entry:
        q_age = age_minutes(entry.get("data_updated_at"), now)
        report["source_health_age_min"] = None if q_age is None else round(q_age, 1)
        if market_active and q_age is not None and q_age > SOURCE_HEALTH_MAX_AGE_MIN:
            report["warnings"].append(
                f"{SOURCE_HEALTH_ASSET} (actiu 24h) no canvia fa {q_age:.0f} min: possible font bloquejada."
            )
    return report


def load_pages_manifest(url, timeout=20):
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(
        f"{url}{sep}_t={int(time.time())}",
        headers={"User-Agent": "dashboard-freshness-watchdog", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def render_markdown(report):
    if report["stale"]:
        state = "RETARD (>60 min en horari de mercat)"
    elif not report["market_active"]:
        state = "OK (fora d'horari de mercat)"
    elif report["needs_self_heal"]:
        state = "AVÍS PREVENTIU (>45 min · auto-reparació)"
    else:
        state = "OK"

    lines = [
        f"## Frescor de dades: {state}",
        "",
        f"- Finestra de mercat activa: `{report['market_active']}` (oberta: `{report['window_open']}`, post-tancament pendent: `{report['post_close_pending']}`)",
        f"- Última consulta efectiva / generació: `{report.get('effective_checked_at')}` (edat efectiva: {report.get('effective_age_min')} min; self-heal {report['self_heal_min']:.0f} min / límit {report['max_age_min']:.0f} min)",
        f"- Última generació (repositori): `{report['repo_generated_at']}` (fa {report['repo_age_min']} min)",
        f"- Publicat a GitHub Pages: `{report.get('pages_generated_at')}`",
        f"- Últim run: event `{report['last_run']['event_name']}`, núm. `{report['last_run']['run_number']}`",
    ]
    for warning in report["warnings"]:
        lines.append(f"- AVÍS: {warning}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="docs/data/manifest.json")
    parser.add_argument("--pages-url", default=DEFAULT_PAGES_URL)
    parser.add_argument("--no-pages", action="store_true")
    parser.add_argument("--max-age-min", type=float, default=DEFAULT_MAX_AGE_MIN)
    parser.add_argument("--self-heal-min", type=float, default=DEFAULT_SELF_HEAL_MIN)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    repo_manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pages_manifest = None if args.no_pages else load_pages_manifest(args.pages_url)
    report = evaluate(
        repo_manifest,
        pages_manifest,
        datetime.now(timezone.utc),
        max_age_min=args.max_age_min,
        self_heal_min=args.self_heal_min,
        check_pages=not args.no_pages,
    )

    text = render_markdown(report)
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(text)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 2 if report["stale"] else 0


if __name__ == "__main__":
    sys.exit(main())
