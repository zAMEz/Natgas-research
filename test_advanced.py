"""Focused temporal-causality and accounting checks for advanced research."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from advanced_data import prepare_features, executable_dataset
from advanced_backtest import (
    simulate, session_days, daily_pnl, metrics, select_policy, reality_check,
)
from train_advanced import Config, split_windows, run_fold, select_research_candidate


def raw_minutes(rows=360):
    """Regular tradable session with distinct auxiliary contracts."""
    i = np.arange(rows)
    prices = 3.0 + i * 0.001
    frame = pd.DataFrame({
        "ts_event": pd.date_range("2026-06-01 12:00", periods=rows, freq="min", tz="UTC"),
        "instrument_id": 101,
        "open": prices,
        "high": prices + 0.002,
        "low": prices - 0.001,
        "close": prices + 0.001,
        "volume": 50 + i % 9,
        "data_condition": "available",
    })
    for j, market in enumerate(("wti", "lng_equity", "ng_m1", "ng_m2", "ng_m3")):
        frame[f"{market}_close"] = 4.0 + j + i * 0.001
        frame[f"{market}_instrument_id"] = 201 + j
        frame[f"{market}_available"] = 1
        frame[f"{market}_age_minutes"] = 0
    frame["eia_minutes_to_release"] = 1000.0
    frame["eia_minutes_since_release"] = 1000.0
    frame["eia_post_release_60m"] = 0
    frame["eia_release_window"] = 0
    return frame


class FeatureAndExecutionTests(unittest.TestCase):
    def test_future_mutation_does_not_change_past_features(self):
        raw = raw_minutes()
        first, names, _ = prepare_features(raw)
        changed = raw.copy()
        later = changed.index >= 180
        for name in ("open", "high", "low", "close", "wti_close", "ng_m2_close"):
            changed.loc[later, name] += 100
        changed.loc[later, "volume"] *= 100
        changed.loc[later, "instrument_id"] = 999
        changed.loc[later, "data_condition"] = "degraded"
        second, second_names, _ = prepare_features(changed)
        self.assertEqual(names, second_names)
        pd.testing.assert_frame_equal(first.loc[:179, names], second.loc[:179, names])

    def test_auxiliary_contract_roll_resets_its_own_returns(self):
        raw = raw_minutes()
        raw.loc[120:, "wti_instrument_id"] = 999
        raw.loc[120:, "wti_close"] += 20
        frame, _, _ = prepare_features(raw)
        self.assertEqual(frame.loc[119, "segment"], frame.loc[120, "segment"])
        self.assertTrue(pd.isna(frame.loc[120, "wti_return_1m"]))
        self.assertTrue(frame.loc[120:124, "wti_return_5m"].isna().all())
        self.assertAlmostEqual(
            frame.loc[125, "wti_return_5m"],
            np.log(raw.loc[125, "wti_close"] / raw.loc[120, "wti_close"]),
            places=7,
        )
        self.assertFalse(pd.isna(frame.loc[120, "lng_equity_return_1m"]))

    def test_auxiliary_missing_data_resets_returns(self):
        raw = raw_minutes()
        raw.loc[120, "wti_available"] = 0
        raw.loc[120, "wti_close"] = np.nan
        frame, _, _ = prepare_features(raw)
        self.assertTrue(frame.loc[120:121, "wti_return_1m"].isna().all())
        self.assertFalse(pd.isna(frame.loc[122, "wti_return_1m"]))

    def test_duplicate_curve_contracts_are_unavailable_and_reset_changes(self):
        raw = raw_minutes()
        raw.loc[120, "ng_m2_instrument_id"] = raw.loc[120, "ng_m1_instrument_id"]
        frame, _, _ = prepare_features(raw)
        self.assertEqual(frame.loc[120, "curve_available"], 0)
        self.assertTrue(pd.isna(frame.loc[120, "ng_m1_ng_m2_spread_ticks"]))
        self.assertTrue(
            frame.loc[120:125, "ng_m1_ng_m2_spread_ticks_change_5m_vol"].isna().all()
        )

    def test_delayed_entry_exit_and_integer_ticks(self):
        frame, _, _ = prepare_features(raw_minutes())
        result = executable_dataset(frame, 3, stride=1)
        target = frame.iloc[100]
        row = result.loc[result.ts_event.eq(target.ts_event)].iloc[0]
        self.assertEqual(row.decision_time, target.ts_event + pd.Timedelta(minutes=1))
        self.assertEqual(row.entry_time, target.ts_event + pd.Timedelta(minutes=2))
        self.assertEqual(row.exit_time, target.ts_event + pd.Timedelta(minutes=5))
        self.assertEqual(row.entry_price, frame.iloc[102].open)
        self.assertEqual(row.exit_price, frame.iloc[105].open)
        self.assertEqual(row.target_ticks, 3)
        np.testing.assert_array_equal(result.target_ticks, np.rint(result.target_ticks))

    def test_target_path_rejects_gap_roll_and_bad_quality(self):
        original = raw_minutes()
        for problem in ("gap", "roll", "quality"):
            with self.subTest(problem=problem):
                raw = original.copy()
                if problem == "gap":
                    raw = raw.drop(index=150)
                elif problem == "roll":
                    raw.loc[150:, "instrument_id"] = 999
                else:
                    raw.loc[150, "data_condition"] = "degraded"
                frame, _, _ = prepare_features(raw)
                result = executable_dataset(frame, 5, stride=1)
                self.assertFalse(result.ts_event.eq(original.loc[147, "ts_event"]).any())
                self.assertTrue(result.ts_event.eq(original.loc[140, "ts_event"]).any())
                self.assertFalse(result.ts_event.eq(original.loc[150, "ts_event"]).any())

    def test_storage_actual_is_excluded_and_decision_grid_is_respected(self):
        raw = raw_minutes()
        raw["storage_change_bcf"] = 100
        frame, names, _ = prepare_features(raw)
        self.assertNotIn("storage_change_bcf", names)
        result = executable_dataset(frame, 5, stride=5)
        self.assertTrue(result.decision_time.dt.minute.mod(5).eq(0).all())

    def test_eia_safety_window_and_same_session(self):
        raw = raw_minutes()
        raw.loc[100, "eia_minutes_to_release"] = 10
        raw.loc[101, "eia_minutes_since_release"] = 10
        frame, _, _ = prepare_features(raw)
        result = executable_dataset(frame, 5, stride=1)
        self.assertFalse(result.ts_event.isin(raw.loc[[100, 101], "ts_event"]).any())
        self.assertTrue(result.ts_event.eq(raw.loc[102, "ts_event"]).any())
        self.assertTrue(result.exit_time.dt.tz_convert("America/New_York").dt.hour.lt(16).all())

    def test_duplicate_timestamps_fail_closed(self):
        raw = raw_minutes()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_features(pd.concat([raw, raw.iloc[[100]]], ignore_index=True))


class SplitAndAccountingTests(unittest.TestCase):
    def test_split_boundaries_purge_actual_exit_times(self):
        times = pd.date_range("2025-10-01", "2026-06-01", freq="30min", tz="UTC")
        data = pd.DataFrame({
            "decision_time": times,
            "entry_time": times + pd.Timedelta(minutes=1),
            "exit_time": times + pd.Timedelta(minutes=30),
        })
        start = pd.Timestamp("2026-05-01", tz="UTC")
        end = pd.Timestamp("2026-06-01", tz="UTC")
        windows = split_windows(data, start, end)
        expected = {
            "train": (pd.Timestamp("2025-09-19", tz="UTC"), pd.Timestamp("2026-03-18", tz="UTC")),
            "stop": (pd.Timestamp("2026-03-18", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
            "calibration": (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-04-15", tz="UTC")),
            "validation": (pd.Timestamp("2026-04-15", tz="UTC"), start),
            "test": (start, end),
        }
        for name, (lower, upper) in expected.items():
            with self.subTest(partition=name):
                values = windows[name]
                self.assertTrue(values.decision_time.ge(lower).all())
                self.assertTrue(values.exit_time.lt(upper).all())
                self.assertFalse(values.decision_time.eq(upper - pd.Timedelta(minutes=30)).any())
                self.assertTrue(values.decision_time.eq(upper - pd.Timedelta(minutes=60)).any())

    def test_nonoverlapping_trades_allow_entry_exactly_at_prior_exit(self):
        frame, _, _ = prepare_features(raw_minutes())
        data = executable_dataset(frame, 5, stride=1).head(20)
        trades = simulate(data, np.full(len(data), 10.0), cost=2, edge=0)
        self.assertEqual(len(trades), 4)
        self.assertTrue(
            (trades.entry_time.iloc[1:].to_numpy() >= trades.exit_time.iloc[:-1].to_numpy()).all()
        )
        self.assertEqual(trades.iloc[1].entry_time, trades.iloc[0].exit_time)
        np.testing.assert_array_equal(trades.net_ticks, trades.gross_ticks - 2)

    def test_drawdown_starts_at_zero_and_counts_zero_trade_days(self):
        days = pd.date_range("2026-06-01", periods=3, freq="D")
        trades = pd.DataFrame({
            "entry_time": pd.to_datetime(["2026-06-01 14:00Z", "2026-06-03 14:00Z"]),
            "net_ticks": [-5.0, 2.0],
        })
        stats = metrics(trades, days)
        self.assertEqual(stats["max_drawdown_ticks"], -5)
        self.assertEqual(stats["evaluated_days"], 3)
        np.testing.assert_array_equal(daily_pnl(trades, days), [-5, 0, 2])
        expected_sharpe = np.sqrt(252) * np.mean([-5, 0, 2]) / np.std([-5, 0, 2], ddof=1)
        self.assertAlmostEqual(stats["daily_sharpe"], expected_sharpe)

    def test_empty_strategy_is_cash(self):
        frame, _, _ = prepare_features(raw_minutes())
        data = executable_dataset(frame, 5, stride=1)
        trades = simulate(data, np.zeros(len(data)), cost=2, edge=0)
        stats = metrics(trades, session_days(data))
        self.assertEqual(stats["trades"], 0)
        self.assertEqual(stats["net_ticks"], 0)
        self.assertEqual(stats["max_drawdown_ticks"], 0)
        policy, _ = select_policy(data, np.zeros(len(data)), cost=2)
        self.assertFalse(policy["enabled"])

    def test_family_test_cash_is_not_significant(self):
        result = reality_check(np.zeros((20, 3)), samples=20)
        self.assertEqual(result["family_p_value"], 1)
        self.assertEqual(result["candidate_count"], 3)

    def test_research_selection_ignores_holdout_winner(self):
        summary = pd.DataFrame([
            {"phase": "development", "model": "ridge", "horizon": 5,
             "cost_ticks": 2, "trades": 50, "mean_daily_ci_low": -1, "net_ticks": 20},
            {"phase": "holdout", "model": "ebm", "horizon": 30,
             "cost_ticks": 2, "trades": 1000, "mean_daily_ci_low": 100, "net_ticks": 10000},
        ])
        self.assertEqual(select_research_candidate(summary, Config())["model"], "cash")

    def test_run_fold_cost_stress_preserves_selected_trades(self):
        """Exercise fold accounting with deterministic forecasts and no fitting."""
        frame, features, _ = prepare_features(raw_minutes())
        data = executable_dataset(frame, 5, stride=1).head(50)
        windows = {name: data.copy() for name in ("train", "stop", "calibration", "validation", "test")}
        model_names = ("lightgbm", "ebm", "random_forest", "extra_trees", "ridge", "ensemble",
                       "lightgbm_q10", "lightgbm_q90")
        predictions = {
            model: {partition: np.full(len(data), 10.0) for partition in windows if partition != "train"}
            for model in model_names
        }
        policy = {"edge": 0.0, "probability_floor": 0.0, "enabled": True, "validation_score": 1}
        with tempfile.TemporaryDirectory(prefix="ng-fold-test-") as directory:
            with (
                patch("train_advanced.split_windows", return_value=windows),
                patch("train_advanced.fit_predict_models", return_value=({}, predictions, [])),
                patch("train_advanced.fit_meta", return_value=(
                    None, {name: np.ones(len(data)) for name in windows if name != "train"}
                )),
                patch("train_advanced.select_policy", return_value=(policy, [])),
            ):
                result = run_fold(
                    data, features, pd.Timestamp("2026-06-01", tz="UTC"),
                    pd.Timestamp("2026-07-01", tz="UTC"), "development", 5,
                    Config(), Path(directory),
                )
        strategies = pd.DataFrame(result["strategies"])
        for _, group in strategies.groupby("model"):
            self.assertEqual(group.trades.nunique(), 1)
            base = group.loc[group.cost_ticks.eq(2)].iloc[0]
            for row in group.itertuples():
                self.assertAlmostEqual(row.net_ticks, base.net_ticks - (row.cost_ticks - 2) * base.trades)


if __name__ == "__main__":
    unittest.main()
