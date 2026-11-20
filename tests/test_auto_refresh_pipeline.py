import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_pages_data as bpd
import data_manager as dm


class AutoRefreshPipelineTests(unittest.TestCase):
    def test_post_close_refresh_captures_missed_close_once_and_caches_weekend(self):
        # Saturday 02:15 UTC (after Friday 16:20 America/New_York close = 21:20 UTC).
        sat_utc = pd.Timestamp("2026-11-21T02:15:00")
        # Previous check was Friday 18:38 UTC (intraday, before NY close) -> must refresh.
        self.assertTrue(
            bpd._needs_post_close_refresh(
                "market",
                {"last_checked_at": "2026-11-20T18:38:00Z"},
                "2026-11-20T18:38:00Z",
                sat_utc,
            )
        )
        # Once Friday close has been checked after 21:20 UTC, weekend runs do not re-download.
        self.assertFalse(
            bpd._needs_post_close_refresh(
                "market",
                {"last_checked_at": "2026-11-20T22:10:00Z"},
                "2026-11-20T22:10:00Z",
                sat_utc,
            )
        )
        # 24h assets (e.g. BTC-USD) never use market post-close override; they follow _is_refresh_allowed.
        self.assertFalse(
            bpd._needs_post_close_refresh(
                "24h",
                {"last_checked_at": "2026-11-20T18:38:00Z"},
                "2026-11-20T18:38:00Z",
                sat_utc,
            )
        )

    def test_data_manager_preserves_valid_csv_and_reports_fallback_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "TEST.csv"
            df_seed = pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [102.0, 103.0],
                    "Low": [99.0, 100.0],
                    "Close": [101.0, 102.5],
                    "Volume": [1000.0, 1200.0],
                },
                index=pd.to_datetime(["2026-11-18", "2026-11-19"]),
            )
            df_seed.index.name = "Date"
            df_seed.to_csv(csv_path)

            with patch.object(dm, "CARPETA_DADES", str(tmp_path)), patch.dict(
                dm._CACHE_MEMORIA, {}, clear=True
            ), patch.dict(dm._LAST_FETCH_INFO, {}, clear=True), patch.object(
                dm, "_refresh_recent_window", side_effect=RuntimeError("Yahoo HTTP 503")
            ):
                result = dm.get_data("TEST", force_refresh=True, allow_stale=False)
                info = dm.get_last_fetch_info("TEST")

            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(float(result["close"].iloc[-1]), 102.5)
            self.assertEqual(info["status"], "fallback_error")
            self.assertTrue(info["attempted_refresh"])
            self.assertEqual(info["last_candle_date"], "2026-11-19")
            self.assertIn("Yahoo HTTP 503", info["error"])

    def test_build_site_outputs_diagnostics_and_preserves_data_updated_at_when_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / "docs" / "data"
            pd.DataFrame(
                [
                    {"Nom": "Nasdaq", "Ticker": "^NDX", "RefreshGroup": "market"},
                    {"Nom": "MSCI_Word", "Ticker": "^990100-USD-STRD", "RefreshGroup": "market"},
                    {"Nom": "VIX", "Ticker": "^VIX", "RefreshGroup": "market"},
                ]
            ).to_csv(base / "llista_tickers.csv", index=False)

            dates = pd.bdate_range("2025-01-01", periods=260)
            sample_df = pd.DataFrame(
                {
                    "open": [100.0 + i * 0.1 for i in range(len(dates))],
                    "high": [101.0 + i * 0.1 for i in range(len(dates))],
                    "low": [99.0 + i * 0.1 for i in range(len(dates))],
                    "close": [100.5 + i * 0.1 for i in range(len(dates))],
                    "volume": [1000.0 for _ in range(len(dates))],
                },
                index=dates,
            )
            sample_df.index.name = "Date"
            sample_rate = pd.Series([2.5, 2.5], index=pd.to_datetime(["2025-01-01", "2026-01-01"]))

            def fake_get_data(ticker, force_refresh=False, allow_stale=False):
                dm._LAST_FETCH_INFO[dm._normalize_ticker(ticker)] = {
                    "ticker": dm._normalize_ticker(ticker),
                    "status": "updated",
                    "attempted_refresh": True,
                    "candle_changed": True,
                    "checked_at": "2026-11-20T18:00:00Z",
                    "last_candle_date": str(dates[-1].date()),
                    "error": None,
                }
                return sample_df.copy()

            with patch.object(bpd, "get_data", side_effect=fake_get_data), patch.object(
                bpd, "get_ecb_deposit_rate", return_value=sample_rate
            ), patch.object(bpd, "get_fed_rate", return_value=sample_rate), patch.object(
                bpd.macro_context, "build_macro_data", return_value={"meta": {"generated_at": "2026-11-20T18:00:00Z"}, "status": {}, "items": [], "matrix": {}}
            ), patch.object(
                bpd, "_local_refresh_time", return_value=pd.Timestamp("2026-11-20T19:00:00+01:00")
            ), patch.dict(
                os.environ, {"WORKFLOW_EXECUTED_AT": "2026-11-20T17:59:50Z", "GITHUB_RUN_NUMBER": "999"}
            ):
                bpd.build_site(base, out, force_refresh=True, refresh_rates=False)

            manifest1 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest1["diagnostics"]["executed_at"], "2026-11-20T17:59:50Z")
            self.assertEqual(manifest1["diagnostics"]["run_number"], "999")
            ndx1 = next(t for t in manifest1["tickers"] if t["slug"] == "nasdaq")
            self.assertEqual(ndx1["quote_date"], str(dates[-1].date()))
            self.assertEqual(ndx1["refresh_status"], "updated")
            first_data_updated_at = ndx1["data_updated_at"]

            # Second run with unchanged candle must keep previous data_updated_at while advancing generated_at
            def fake_get_data_unchanged(ticker, force_refresh=False, allow_stale=False):
                dm._LAST_FETCH_INFO[dm._normalize_ticker(ticker)] = {
                    "ticker": dm._normalize_ticker(ticker),
                    "status": "unchanged",
                    "attempted_refresh": True,
                    "candle_changed": False,
                    "checked_at": "2026-11-20T18:30:00Z",
                    "last_candle_date": str(dates[-1].date()),
                    "error": None,
                }
                return sample_df.copy()

            orig_utcnow = pd.Timestamp.utcnow
            with patch.object(bpd, "get_data", side_effect=fake_get_data_unchanged), patch.object(
                bpd, "get_ecb_deposit_rate", return_value=sample_rate
            ), patch.object(bpd, "get_fed_rate", return_value=sample_rate), patch.object(
                bpd.macro_context, "build_macro_data", return_value={"meta": {"generated_at": "2026-11-20T18:30:00Z"}, "status": {}, "items": [], "matrix": {}}
            ), patch.object(
                bpd, "_local_refresh_time", return_value=pd.Timestamp("2026-11-20T19:30:00+01:00")
            ), patch.object(
                pd.Timestamp, "utcnow", side_effect=lambda: orig_utcnow() + pd.Timedelta(minutes=30)
            ):
                bpd.build_site(base, out, force_refresh=True, refresh_rates=False)

            manifest2 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            ndx2 = next(t for t in manifest2["tickers"] if t["slug"] == "nasdaq")
            self.assertNotEqual(manifest2["generated_at"], manifest1["generated_at"])
            self.assertEqual(ndx2["refresh_status"], "unchanged")
            self.assertEqual(ndx2["data_updated_at"], first_data_updated_at)

    def test_frontend_implements_auto_refresh_and_diagnostic_status(self):
        app_js = (Path(__file__).resolve().parent.parent / "docs" / "app.js").read_text(encoding="utf-8")
        for token in [
            "startAutoRefresh",
            "checkForUpdates",
            "visibilitychange",
            "refreshInFlight",
            "tickerRequestSeq",
            "preserveViewport",
            "updateStatusDiagnostic",
            "last-modified",
            "withCacheBust",
            "STALE_AFTER_MS",
            "Últim canvi de preu",
            "Última consulta font",
            "Última execució",
            "més de 60 min",
        ]:
            self.assertIn(token, app_js)


if __name__ == "__main__":
    unittest.main()
