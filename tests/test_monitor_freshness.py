import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import monitor_freshness as mf

# Friday 12:00 UTC = 13:00 Europe/Madrid (inside active market hours 07-23)
NOW_MARKET = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
# Saturday 12:00 UTC = 13:00 Europe/Madrid (weekend after Friday close captured)
NOW_WEEKEND = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
# Friday 06:25 UTC = 07:25 Europe/Madrid (25 minutes after morning window opens)
NOW_MORNING_OPEN = datetime(2026, 11, 20, 6, 25, tzinfo=timezone.utc)


def iso(minutes_ago, ref=NOW_MARKET):
    return (ref - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest(
    gen_ago,
    *,
    checked_ago=None,
    btc_changed_ago=5,
    ndx_changed_ago=900,
    ref=NOW_MARKET,
    error_count=0,
    updated_count=0,
    unchanged_count=2,
):
    chk = gen_ago if checked_ago is None else checked_ago
    return {
        "generated_at": iso(gen_ago, ref),
        "diagnostics": {
            "event_name": "schedule",
            "run_number": "1",
            "executed_at": iso(gen_ago, ref),
            "market_window_open": True,
            "error_count": error_count,
            "updated_count": updated_count,
            "unchanged_count": unchanged_count,
        },
        "tickers": [
            {
                "ticker": "BTC-USD",
                "refresh_group": "24h",
                "data_updated_at": iso(btc_changed_ago, ref),
                "last_checked_at": iso(chk, ref),
            },
            {
                "ticker": "^NDX",
                "refresh_group": "market",
                "data_updated_at": iso(ndx_changed_ago, ref),
                "last_checked_at": iso(chk, ref),
            },
        ],
    }


class FreshnessTests(unittest.TestCase):
    def test_unchanged_quote_with_recent_source_check_is_healthy_and_quiet(self):
        # Market open: ^NDX quote unchanged for 15h, but effective check was 20 min ago.
        report = mf.evaluate(manifest(20, ndx_changed_ago=900), manifest(20), NOW_MARKET)
        self.assertTrue(report["market_active"])
        self.assertFalse(report["needs_self_heal"])
        self.assertFalse(report["stale"])
        self.assertEqual(report["warnings"], [])

    def test_45_to_60_min_delay_triggers_silent_self_heal_without_error(self):
        # 52 minutes old: tolerates up to 60 min without breaching (stale=False),
        # while quietly requesting preventive self-heal (>45 min).
        report = mf.evaluate(manifest(52), manifest(52), NOW_MARKET)
        self.assertFalse(report["stale"])
        self.assertTrue(report["needs_self_heal"])

    def test_over_60_min_in_market_hours_marks_stale(self):
        report = mf.evaluate(manifest(65), manifest(65), NOW_MARKET)
        self.assertTrue(report["stale"])
        self.assertTrue(report["needs_self_heal"])

    def test_outside_market_hours_after_friday_close_is_never_stale(self):
        # Friday post-close captured at 21:30Z; Saturday 12:00Z is 14.5h later -> quiet OK.
        fri_close_manifest = {
            "generated_at": "2026-11-20T21:30:00Z",
            "diagnostics": {"event_name": "schedule", "run_number": "1", "executed_at": "2026-11-20T21:30:00Z"},
            "tickers": [
                {
                    "ticker": "^NDX",
                    "refresh_group": "market",
                    "data_updated_at": "2026-11-20T21:30:00Z",
                    "last_checked_at": "2026-11-20T21:30:00Z",
                },
                {
                    "ticker": "BTC-USD",
                    "refresh_group": "24h",
                    "data_updated_at": "2026-11-20T21:30:00Z",
                    "last_checked_at": "2026-11-20T21:30:00Z",
                },
            ],
        }
        report = mf.evaluate(fri_close_manifest, fri_close_manifest, NOW_WEEKEND)
        self.assertFalse(report["market_active"])
        self.assertFalse(report["stale"])
        self.assertFalse(report["needs_self_heal"])
        self.assertEqual(report["warnings"], [])

    def test_morning_open_counts_only_elapsed_time_since_window_open(self):
        # Prior close captured at Thursday 21:30Z (= 22:30 Madrid). At Friday 07:25 Madrid,
        # effective delay inside market hours is only 25 min, not ~9 hours.
        thu_close_manifest = {
            "generated_at": "2026-11-19T21:30:00Z",
            "diagnostics": {"event_name": "schedule", "run_number": "1", "executed_at": "2026-11-19T21:30:00Z"},
            "tickers": [
                {
                    "ticker": "^NDX",
                    "refresh_group": "market",
                    "data_updated_at": "2026-11-19T21:30:00Z",
                    "last_checked_at": "2026-11-19T21:30:00Z",
                }
            ],
        }
        report = mf.evaluate(thu_close_manifest, thu_close_manifest, NOW_MORNING_OPEN)
        self.assertTrue(report["market_active"])
        self.assertAlmostEqual(report["effective_age_min"], 25.0, places=1)
        self.assertFalse(report["stale"])
        self.assertFalse(report["needs_self_heal"])

    def test_missing_heartbeat_is_stale(self):
        report = mf.evaluate({}, None, NOW_MARKET)
        self.assertTrue(report["stale"])
        self.assertTrue(report["needs_self_heal"])

    def test_recent_upstream_source_failure_warns_without_false_scheduler_alarm(self):
        m = manifest(10, checked_ago=95, error_count=2, updated_count=0, unchanged_count=0)
        report = mf.evaluate(m, m, NOW_MARKET)
        self.assertFalse(report["stale"])
        self.assertTrue(any("Font financera" in w for w in report["warnings"]))

    def test_pages_lagging_behind_repository_is_flagged_only_after_grace(self):
        lagging = mf.evaluate(manifest(30), manifest(60), NOW_MARKET)
        self.assertTrue(lagging.get("pages_stale"))
        deploying = mf.evaluate(manifest(5), manifest(35), NOW_MARKET)
        self.assertFalse(deploying.get("pages_stale", False))

    def test_frozen_24h_asset_warns_without_failing(self):
        report = mf.evaluate(manifest(10, btc_changed_ago=400), manifest(10), NOW_MARKET)
        self.assertFalse(report["stale"])
        self.assertTrue(any("BTC-USD" in w for w in report["warnings"]))

    def test_cli_exit_codes_and_json_output_respect_60m_market_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            m_path = Path(tmp) / "manifest.json"
            out_path = Path(tmp) / "out.json"
            # Mock datetime.now inside monitor_freshness to NOW_MARKET so test is clock-deterministic.
            fresh_ts = iso(20, NOW_MARKET)
            tolerated_ts = iso(50, NOW_MARKET)
            stale_ts = iso(75, NOW_MARKET)

            class FixedDateTime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return NOW_MARKET if tz else NOW_MARKET.replace(tzinfo=None)

            with patch.object(mf, "datetime", FixedDateTime), patch.dict("os.environ", {}, clear=True):
                m_path.write_text(json.dumps({"generated_at": fresh_ts, "tickers": []}), encoding="utf-8")
                self.assertEqual(mf.main(["--manifest", str(m_path), "--no-pages", "--json-out", str(out_path)]), 0)
                self.assertFalse(json.loads(out_path.read_text(encoding="utf-8"))["stale"])

                m_path.write_text(json.dumps({"generated_at": tolerated_ts, "tickers": []}), encoding="utf-8")
                self.assertEqual(mf.main(["--manifest", str(m_path), "--no-pages", "--json-out", str(out_path)]), 0)
                res_tol = json.loads(out_path.read_text(encoding="utf-8"))
                self.assertFalse(res_tol["stale"])
                self.assertTrue(res_tol["needs_self_heal"])

                m_path.write_text(json.dumps({"generated_at": stale_ts, "tickers": []}), encoding="utf-8")
                self.assertEqual(mf.main(["--manifest", str(m_path), "--no-pages", "--json-out", str(out_path)]), 2)
                self.assertTrue(json.loads(out_path.read_text(encoding="utf-8"))["stale"])

    def test_workflows_match_60m_market_hours_criteria_and_no_duplicate_comments(self):
        root = Path(__file__).resolve().parent.parent
        watch_text = (root / ".github" / "workflows" / "watch-data-freshness.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "22,52 6-22 * * 1-5"', watch_text)
        self.assertIn("group: watch-dashboard-freshness", watch_text)
        self.assertIn("cancel-in-progress: false", watch_text)
        self.assertIn('"$AGE" -ge 120', watch_text)
        self.assertIn("gh issue edit", watch_text)
        self.assertNotIn('gh issue comment "$OPEN"', watch_text)

        update_text = (root / ".github" / "workflows" / "update-pages-data.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "7,37 6-22 * * 1-5"', update_text)
        self.assertIn('cron: "12 3 * * *"', update_text)
        self.assertIn("cancel-in-progress: false", update_text)
        self.assertIn("refresh_rates", update_text)


if __name__ == "__main__":
    unittest.main()
