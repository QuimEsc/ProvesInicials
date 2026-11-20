#!/usr/bin/env python3
"""Freshness watchdog for the dashboard data pipeline.

It measures the *system heartbeat* (``generated_at`` of docs/data/manifest.json,
rewritten by every successful run even if no price changes) and never the age of
a quotation. A closed market therefore cannot raise a false alarm.

Exit codes: 0 = healthy, 2 = heartbeat older than the threshold.
Pure standard library; no market data and no model code is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MAX_AGE_MIN = 45.0
DEFAULT_PAGES_LAG_MIN = 20.0
DEFAULT_PAGES_URL = "https://quimesc.github.io/ProvesInicials/docs/data/manifest.json"
SOURCE_HEALTH_ASSET = "BTC-USD"  # 24h asset: its price changes continuously
SOURCE_HEALTH_MAX_AGE_MIN = 180.0


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
    ts = parse_utc(value)
    if ts is None:
        return None
    return (now - ts).total_seconds() / 60.0


def evaluate(repo_manifest, pages_manifest, now, *, max_age_min=DEFAULT_MAX_AGE_MIN,
             pages_lag_min=DEFAULT_PAGES_LAG_MIN):
    """Return a report dict. ``stale`` only depends on the pipeline heartbeat."""
    repo_age = age_minutes(repo_manifest.get("generated_at"), now)
    report = {
        "now": now.isoformat(timespec="seconds"),
        "max_age_min": max_age_min,
        "repo_generated_at": repo_manifest.get("generated_at"),
        "repo_age_min": None if repo_age is None else round(repo_age, 1),
        "stale": repo_age is None or repo_age > max_age_min,
        "warnings": [],
    }
    diag = repo_manifest.get("diagnostics") or {}
    report["last_run"] = {
        "event_name": diag.get("event_name"),
        "run_number": diag.get("run_number"),
        "executed_at": diag.get("executed_at"),
    }

    if pages_manifest is None:
        report["pages_generated_at"] = None
        report["warnings"].append("No s'ha pogut llegir el manifest publicat a GitHub Pages.")
    else:
        pages_gen = pages_manifest.get("generated_at")
        report["pages_generated_at"] = pages_gen
        repo_ts, pages_ts = parse_utc(repo_manifest.get("generated_at")), parse_utc(pages_gen)
        if repo_ts and pages_ts and repo_ts > pages_ts:
            lag = (repo_ts - pages_ts).total_seconds() / 60.0
            # Only meaningful once the deployment had time to finish.
            if repo_age is not None and repo_age > pages_lag_min:
                report["pages_stale"] = True
                report["warnings"].append(
                    f"GitHub Pages serveix dades {lag:.0f} min més antigues que el repositori."
                )

    entry = next((t for t in repo_manifest.get("tickers", []) if t.get("ticker") == SOURCE_HEALTH_ASSET), None)
    if entry:
        q_age = age_minutes(entry.get("data_updated_at"), now)
        report["source_health_age_min"] = None if q_age is None else round(q_age, 1)
        if q_age is not None and q_age > SOURCE_HEALTH_MAX_AGE_MIN:
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
    state = "RETARD" if report["stale"] else "OK"
    lines = [
        f"## Frescor de dades: {state}",
        "",
        f"- Última generació (repositori): `{report['repo_generated_at']}` (fa {report['repo_age_min']} min, límit {report['max_age_min']:.0f} min)",
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
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    repo_manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pages_manifest = None if args.no_pages else load_pages_manifest(args.pages_url)
    report = evaluate(repo_manifest, pages_manifest, datetime.now(timezone.utc), max_age_min=args.max_age_min)

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
