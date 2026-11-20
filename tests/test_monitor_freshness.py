import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import monitor_freshness as mf

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def iso(minutes_ago):
    return (NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest(gen_ago, btc_changed_ago=5, ndx_changed_ago=900):
    return {
        "generated_at": iso(gen_ago),
        "diagnostics": {"event_name": "schedule", "run_number": "1", "executed_at": iso(gen_ago)},
        "tickers": [
            {"ticker": "BTC-USD", "data_updated_at": iso(btc_changed_ago)},
            {"ticker": "^NDX", "data_updated_at": iso(ndx_changed_ago)},
        ],
    }


class FreshnessTests(unittest.TestCase):
    def test_closed_market_with_unchanged_quote_is_not_an_alarm(self):
        # Nasdaq quote unchanged for 15 hours, but the pipeline heartbeat is 12 min old.
        report = mf.evaluate(manifest(12, ndx_changed_ago=900), manifest(12), NOW)
        self.assertFalse(report["stale"])
        self.assertEqual(report["warnings"], [])

    def test_one_missed_slot_is_tolerated(self):
        # Worst normal case: next run still pending, heartbeat ~36 minutes old.
        self.assertFalse(mf.evaluate(manifest(36), manifest(36), NOW)["stale"])

    def test_two_missed_slots_raise_alarm(self):
        self.assertTrue(mf.evaluate(manifest(66), manifest(66), NOW)["stale"])
        self.assertTrue(mf.evaluate(manifest(46), manifest(46), NOW)["stale"])

    def test_missing_heartbeat_is_stale(self):
        self.assertTrue(mf.evaluate({}, None, NOW)["stale"])

    def test_pages_lagging_behind_repository_is_flagged_only_after_grace(self):
        lagging = mf.evaluate(manifest(30), manifest(60), NOW)
        self.assertTrue(lagging.get("pages_stale"))
        deploying = mf.evaluate(manifest(5), manifest(35), NOW)
        self.assertFalse(deploying.get("pages_stale", False))

    def test_frozen_24h_asset_warns_without_failing(self):
        report = mf.evaluate(manifest(10, btc_changed_ago=400), manifest(10), NOW)
        self.assertFalse(report["stale"])
        self.assertTrue(any("BTC-USD" in w for w in report["warnings"]))

    def test_report_is_json_serialisable(self):
        json.dumps(mf.evaluate(manifest(10), manifest(10), NOW))

    def test_cli_exit_codes_and_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            m_path = Path(tmp) / "manifest.json"
            out_path = Path(tmp) / "out.json"
            now_utc = datetime.now(timezone.utc)
            fresh_ts = (now_utc - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
            stale_ts = (now_utc - timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%SZ")

            m_path.write_text(json.dumps({"generated_at": fresh_ts, "tickers": []}), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                rc_ok = mf.main(["--manifest", str(m_path), "--no-pages", "--json-out", str(out_path)])
            self.assertEqual(rc_ok, 0)
            self.assertFalse(json.loads(out_path.read_text(encoding="utf-8"))["stale"])

            m_path.write_text(json.dumps({"generated_at": stale_ts, "tickers": []}), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                rc_stale = mf.main(["--manifest", str(m_path), "--no-pages", "--json-out", str(out_path)])
            self.assertEqual(rc_stale, 2)
            self.assertTrue(json.loads(out_path.read_text(encoding="utf-8"))["stale"])

    def test_watchdog_workflow_yaml_avoids_duplicate_comments(self):
        wf_path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "watch-data-freshness.yml"
        wf_text = wf_path.read_text(encoding="utf-8")
        self.assertIn("group: watch-dashboard-freshness", wf_text)
        self.assertIn("cancel-in-progress: false", wf_text)
        self.assertIn("gh issue edit", wf_text)
        self.assertNotIn('gh issue comment "$OPEN"', wf_text)


if __name__ == "__main__":
    unittest.main()
