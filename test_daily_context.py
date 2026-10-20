"""Small deterministic checks of daily indicators and their information timing."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from daily_context import (
    TZ, _confirmed_divergence, _historical_features, _linked_index, _wilder_rsi,
    add_daily_context,
)
from daily_models import (
    DAILY_SCORE_FEATURES, fit_daily_model, fit_reprojection,
    join_daily_forecasts, prequential_daily_forecasts,
)


def synthetic_sessions(days: int = 36, start: str = "2026-01-05") -> pd.DataFrame:
    """480 exact minute bars per NY session, with overnight gaps and WTI."""
    dates = pd.bdate_range(start, periods=days)
    minute = np.tile(np.arange(480), days)
    day = np.repeat(np.arange(days), 480)
    opens = 3.0 + day * .004 + .03 * np.sin(day / 3)
    closes = opens + (.012 * np.cos(day / 2)) * (minute + 1) / 480
    times = pd.DatetimeIndex(np.concatenate([
        pd.date_range(date + pd.Timedelta(hours=8), periods=480,
                      freq="min", tz=TZ).tz_convert("UTC").as_unit("ns").asi8
        for date in dates
    ]), tz="UTC")
    return pd.DataFrame({
        "ts_event": times,
        "decision_time": times + pd.Timedelta(minutes=1),
        "instrument_id": 101,
        "open": opens,
        "high": np.maximum(opens, closes) + .002,
        "low": np.minimum(opens, closes) - .002,
        "close": closes,
        "volume": 50 + minute % 7,
        "data_condition": "available",
        "wti_close": 65 + .08 * day + .003 * minute,
        "wti_instrument_id": 201,
        "wti_available": 1,
        "wti_age_minutes": 0,
    })


def synthetic_daily(closes: np.ndarray) -> pd.DataFrame:
    """Minimal complete daily records for independent indicator checks."""
    count = len(closes)
    opens = np.r_[closes[0], closes[:-1]]
    frame = pd.DataFrame({
        "valid": True, "instrument_id": 101,
        "open": opens, "close": closes,
        "high": np.maximum(opens, closes) + .01,
        "low": np.minimum(opens, closes) - .01,
        "wti_valid": True, "wti_instrument_id": 201,
        "wti_open": np.full(count, 65.0), "wti_close": np.full(count, 65.0),
        "wti_high": np.full(count, 65.1), "wti_low": np.full(count, 64.9),
        "coverage": 1.,
    })
    _linked_index(frame)
    _linked_index(frame, "wti_")
    frame["daily_log_return"] = np.log(frame.close / frame.open)
    frame["daily_range_log"] = np.log(frame.high / frame.low)
    frame["daily_return_ticks"] = (frame.close - frame.open) / .001
    frame["daily_range_ticks"] = (frame.high - frame.low) / .001
    return frame


class DailyContextCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = synthetic_sessions()
        cls.enriched, cls.names, cls.daily, cls.audit = add_daily_context(cls.raw)
        cls.final_date = cls.daily.session_date.iloc[-1]
        cls.final_start = cls.daily.snapshot_at.iloc[-1]

    def test_static_assessment_is_frozen_and_uses_previous_session(self):
        daily = self.daily
        final = self.enriched.loc[self.enriched.ts_event.ge(self.final_start)]
        static = [name for name in self.names if name in daily]
        self.assertTrue(final[static].nunique(dropna=False).eq(1).all())
        for name in static:
            if pd.notna(daily.iloc[-1][name]):
                self.assertAlmostEqual(
                    float(final.iloc[0][name]), float(daily.iloc[-1][name]), places=5
                )
            else:
                self.assertTrue(final[name].isna().all())
        self.assertAlmostEqual(
            final.ctx_prev_day_return.iloc[0],
            np.log(daily.close.iloc[-2] / daily.open.iloc[-2]), places=7,
        )
        self.assertEqual(
            daily.previous_session_end.iloc[-1], daily.available_at.iloc[-2]
        )
        self.assertLess(daily.previous_session_end.iloc[-1], self.final_start)

    def test_later_price_quality_roll_and_missing_minute_cannot_rewrite_morning(self):
        changed = self.raw.copy()
        cutoff = self.final_start + pd.Timedelta(hours=3)
        future = changed.ts_event.ge(cutoff)
        for name in ("open", "high", "low", "close", "wti_close"):
            changed.loc[future, name] *= 4
        changed.loc[future, "data_condition"] = "degraded"
        changed.loc[future, "instrument_id"] = 999
        changed.loc[future, "wti_instrument_id"] = 999
        changed = changed.drop(changed.index[-10]).reset_index(drop=True)
        altered, names, daily, _ = add_daily_context(changed)
        self.assertEqual(names, self.names)
        earlier = self.enriched.ts_event.lt(cutoff)
        other_earlier = altered.ts_event.lt(cutoff)
        pd.testing.assert_frame_equal(
            self.enriched.loc[earlier, names].reset_index(drop=True),
            altered.loc[other_earlier, names].reset_index(drop=True),
        )
        self.assertFalse(daily.valid.iloc[-1])
        self.assertTrue(pd.isna(daily.future_1d_log_return.iloc[-1]))
        self.assertTrue(self.daily.valid.iloc[-1])

    def test_truncating_current_session_preserves_all_available_features(self):
        cutoff = self.final_start + pd.Timedelta(hours=3)
        prefix = self.raw.loc[self.raw.ts_event.lt(cutoff)].copy()
        shortened, names, daily, _ = add_daily_context(prefix)
        pd.testing.assert_frame_equal(
            self.enriched.loc[self.enriched.ts_event.lt(cutoff), names].reset_index(drop=True),
            shortened[names].reset_index(drop=True),
        )
        self.assertFalse(daily.valid.iloc[-1])
        self.assertTrue(pd.isna(daily.future_1d_available_at.iloc[-1]))
        self.assertTrue(shortened.ctx_today_range_log.iloc[-1] > 0)

    def test_new_snapshot_appears_at_eight_not_before(self):
        prefix = self.raw.loc[self.raw.ts_event.lt(self.final_start)]
        extra = self.raw.loc[self.raw.ts_event.eq(self.final_start)].copy()
        preopen = pd.concat([extra, extra], ignore_index=True)
        preopen["ts_event"] = pd.DatetimeIndex([
            self.final_start - pd.Timedelta(minutes=2),
            self.final_start - pd.Timedelta(minutes=1),
        ])
        preopen["decision_time"] = preopen.ts_event + pd.Timedelta(minutes=1)
        frame = pd.concat([prefix, preopen, self.raw.loc[
            self.raw.ts_event.ge(self.final_start)
        ]], ignore_index=True)
        actual, _, daily, _ = add_daily_context(frame)
        before = actual.loc[actual.decision_time.eq(
            self.final_start - pd.Timedelta(minutes=1)
        )].iloc[0]
        at_open = actual.loc[actual.decision_time.eq(self.final_start)].iloc[0]
        self.assertAlmostEqual(
            before.ctx_prev_day_return, daily.ctx_prev_day_return.iloc[-2], places=7
        )
        self.assertAlmostEqual(
            at_open.ctx_prev_day_return, daily.ctx_prev_day_return.iloc[-1], places=7
        )
        self.assertEqual(at_open.ctx_snapshot_age_hours, 0)
        self.assertTrue(pd.isna(at_open.ctx_today_open_close_log_return))
        self.assertTrue(pd.isna(at_open.ctx_today_range_log))

    def test_today_progress_uses_only_observed_bars(self):
        data = self.enriched.loc[self.enriched.ts_event.ge(self.final_start)]
        opening = self.raw.loc[self.raw.ts_event.eq(self.final_start), "open"].iloc[0]
        row = data.iloc[100]
        observed = data.iloc[:101]
        self.assertAlmostEqual(
            row.ctx_today_open_close_log_return, np.log(row.close / opening), places=7
        )
        self.assertAlmostEqual(
            row.ctx_today_range_log, np.log(observed.high.max() / observed.low.min()),
            places=7,
        )
        self.assertAlmostEqual(row.ctx_session_elapsed_fraction, 101 / 480, places=7)

    def test_labels_are_separate_and_finish_at_actual_future_session(self):
        for name in self.names:
            self.assertTrue(name.startswith("ctx_"))
            self.assertNotIn("future", name)
        self.assertFalse(any(name.startswith("future_") for name in self.enriched))
        daily = self.daily
        self.assertTrue(daily.future_1d_available_at.eq(daily.session_end_time).all())
        self.assertTrue(daily.future_3d_available_at.iloc[:-2].eq(
            daily.session_end_time.shift(-2).iloc[:-2]
        ).all())
        self.assertTrue(daily.future_3d_available_at.iloc[-2:].isna().all())
        self.assertTrue(daily.future_3d_log_return.iloc[-2:].isna().all())
        friday = daily.loc[daily.session_date.dt.dayofweek.eq(4)].iloc[0]
        self.assertEqual(
            friday.future_3d_available_at.tz_convert(TZ).dayofweek, 1
        )
        self.assertGreater(friday.future_3d_available_at, friday.snapshot_at)

    def test_incomplete_previous_session_does_not_forward_fill_stale_trend(self):
        raw = synthetic_sessions(days=25)
        local = raw.ts_event.dt.tz_convert(TZ)
        damaged_day = pd.bdate_range("2026-01-05", periods=25)[-2]
        remove = local.dt.tz_localize(None).dt.normalize().eq(damaged_day) & local.dt.hour.ge(12)
        raw = raw.loc[~remove].reset_index(drop=True)
        _, _, daily, _ = add_daily_context(raw)
        self.assertFalse(daily.valid.iloc[-2])
        self.assertEqual(daily.ctx_prev_session_valid.iloc[-1], 0)
        self.assertTrue(pd.isna(daily.ctx_prev_day_return.iloc[-1]))
        self.assertTrue(pd.isna(daily.ctx_bb20_z.iloc[-1]))
        self.assertTrue(daily.future_3d_log_return.iloc[-4:-1].isna().all())

    def test_entire_missing_weekday_breaks_bridge_and_next_snapshot(self):
        raw = synthetic_sessions(days=25)
        missing_date = pd.bdate_range("2026-01-05", periods=25)[-2]
        dates = raw.ts_event.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
        raw = raw.loc[dates.ne(missing_date)].reset_index(drop=True)
        _, _, daily, _ = add_daily_context(raw)
        missing = daily.loc[daily.session_date.eq(missing_date)].iloc[0]
        self.assertEqual(missing.bars, 0)
        self.assertFalse(missing.valid)
        self.assertTrue(daily.bridge_omitted.iloc[-1])
        self.assertTrue(pd.isna(daily.ctx_prev_day_return.iloc[-1]))
        self.assertEqual(daily.ctx_prev_session_valid.iloc[-1], 0)

    def test_known_calendar_is_available_without_complete_current_day(self):
        raw = synthetic_sessions(days=2, start="2026-06-01").iloc[:-350]
        enriched, _, daily, _ = add_daily_context(raw)
        self.assertTrue(daily.ctx_cooling_season.eq(1).all())
        self.assertTrue(daily.ctx_heating_season.eq(0).all())
        self.assertAlmostEqual(
            daily.ctx_year_sin.iloc[-1],
            np.sin(2 * np.pi * (daily.session_date.iloc[-1].dayofyear - 1) / 365.2425),
        )
        self.assertEqual(enriched.ctx_cooling_season.iloc[-1], 1)

    def test_snapshots_keep_eight_am_local_across_daylight_saving(self):
        _, _, daily, _ = add_daily_context(synthetic_sessions(days=6, start="2026-03-05"))
        self.assertTrue(daily.snapshot_at.dt.tz_convert(TZ).dt.hour.eq(8).all())
        self.assertEqual(daily.snapshot_at.iloc[0].hour, 13)
        self.assertEqual(daily.snapshot_at.iloc[-1].hour, 12)
        self.assertTrue(daily.available_at.dt.tz_convert(TZ).dt.hour.eq(16).all())

    def test_unfinished_or_duplicate_bars_are_rejected(self):
        raw = self.raw.head(500).copy()
        raw.loc[0, "decision_time"] = raw.loc[0, "ts_event"]
        with self.assertRaisesRegex(ValueError, "completion"):
            add_daily_context(raw)
        duplicate = pd.concat([self.raw.head(1), self.raw.head(500)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            add_daily_context(duplicate)


class IndicatorAndRollTests(unittest.TestCase):
    def test_rsi_monotone_flat_and_wilder_initialisation(self):
        for values, expected in (
            (np.arange(30, dtype=float), 100),
            (-np.arange(30, dtype=float), 0),
            (np.full(30, 3.0), 50),
        ):
            rsi = _wilder_rsi(pd.Series(values))
            self.assertTrue(rsi.iloc[:14].isna().all())
            np.testing.assert_allclose(rsi.iloc[14:], expected)
        changes = np.array([1, -2, 3, -1, 2, -4, 1, 2, -1, 2, -1, 1, -2, 3, -1])
        values = pd.Series(np.r_[100, 100 + changes.cumsum()])
        rsi = _wilder_rsi(values)
        gain = np.maximum(changes[:14], 0).mean()
        loss = np.maximum(-changes[:14], 0).mean()
        self.assertAlmostEqual(rsi.iloc[14], 100 - 100 / (1 + gain / loss))
        gain = gain * 13 / 14
        loss = (loss * 13 + 1) / 14
        self.assertAlmostEqual(rsi.iloc[15], 100 - 100 / (1 + gain / loss))

    def test_bollinger_matches_trailing_twenty_completed_index_levels(self):
        closes = 3 + np.arange(65) * .01
        daily = synthetic_daily(closes)
        history = _historical_features(daily)
        levels = np.exp(daily.log_index_close)
        last20 = levels.iloc[-20:]
        z = (levels.iloc[-1] - last20.mean()) / last20.std(ddof=1)
        self.assertAlmostEqual(history.bb20_z.iloc[-1], z)
        self.assertAlmostEqual(history.bb20_position.iloc[-1], .5 + z / 4)
        self.assertAlmostEqual(history.bb20_bandwidth.iloc[-1],
                               4 * last20.std(ddof=1) / last20.mean())
        self.assertGreater(history.days_above_below_ma20.iloc[-1], 20)
        self.assertEqual(history.rsi14.iloc[-1], 100)
        self.assertTrue(history.bb20_z.iloc[:19].isna().all())
        self.assertAlmostEqual(history.return_60d.iloc[-1], np.log(closes[-1] / closes[-61]))

    def test_divergence_requires_second_right_bar_and_never_backfills(self):
        lows = pd.Series([5., 4., 2., 4., 5., 4., 1., 4., 5.])
        low_rsi = pd.Series([50., 45., 20., 40., 50., 45., 30., 45., 50.])
        highs = -lows
        high_rsi = 100 - low_rsi
        for price, rsi, column in (
            (lows, low_rsi, "rsi_bull_div_confirmed"),
            (highs, high_rsi, "rsi_bear_div_confirmed"),
        ):
            full = _confirmed_divergence(price, rsi)
            self.assertEqual(full[column].sum(), 1)
            self.assertEqual(full[column].iloc[6], 0)
            self.assertEqual(full[column].iloc[7], 0)
            self.assertEqual(full[column].iloc[8], 1)
            truncated = _confirmed_divergence(price.iloc[:8], rsi.iloc[:8])
            pd.testing.assert_frame_equal(full.iloc[:8], truncated)
            altered = price.copy()
            altered.iloc[8] = price.iloc[6]
            pd.testing.assert_frame_equal(
                full.iloc[:8], _confirmed_divergence(altered, rsi).iloc[:8]
            )
            self.assertEqual(_confirmed_divergence(altered, rsi)[column].iloc[8], 0)

    def test_ng_and_wti_roll_jumps_do_not_create_price_trends(self):
        daily = synthetic_daily(np.full(30, 3.0))
        daily.loc[15:, "instrument_id"] = 102
        daily.loc[15:, ["open", "high", "low", "close"]] += 10
        daily.loc[10:, "wti_instrument_id"] = 202
        daily.loc[10:, ["wti_open", "wti_high", "wti_low", "wti_close"]] += 30
        _linked_index(daily)
        _linked_index(daily, "wti_")
        np.testing.assert_allclose(daily.log_index_close, 0)
        np.testing.assert_allclose(daily.wti_log_index_close, 0)
        self.assertEqual(daily.roll_boundary.sum(), 1)
        self.assertTrue(daily.roll_boundary.iloc[15])
        self.assertTrue(daily.wti_roll_boundary.iloc[10])
        self.assertFalse(daily.roll_boundary.iloc[10])
        history = _historical_features(daily)
        np.testing.assert_allclose(history.return_5d.dropna(), 0)
        np.testing.assert_allclose(history.ng_wti_relative_return_5d.dropna(), 0)
        np.testing.assert_allclose(history.rsi14.dropna(), 50)

    def test_intraday_contract_switch_cannot_create_a_tradable_day_return(self):
        raw = synthetic_sessions(days=3)
        switch = 480 + 240
        raw.loc[switch:959, "instrument_id"] = 999
        raw.loc[switch:959, ["open", "high", "low", "close"]] += 10
        enriched, _, daily, _ = add_daily_context(raw)
        self.assertFalse(daily.valid.iloc[1])
        self.assertTrue(daily.roll_inside_session.iloc[1])
        self.assertTrue(pd.isna(daily.future_1d_log_return.iloc[1]))
        self.assertTrue(enriched.ctx_today_open_close_log_return.iloc[switch:960].isna().all())
        self.assertTrue(pd.notna(enriched.ctx_today_open_close_log_return.iloc[switch - 1]))
        self.assertEqual(enriched.ctx_today_roll_observed.iloc[switch], 1)


class DailyModelTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.bdate_range("2026-05-04", "2026-09-30")
        index = np.arange(len(dates), dtype=float)
        opens = (dates + pd.Timedelta(hours=8)).tz_localize(TZ).tz_convert("UTC")
        ends = (dates + pd.Timedelta(hours=16)).tz_localize(TZ).tz_convert("UTC")
        labels = pd.Series(.01 * np.sin(index / 2) + .002 * np.cos(index / 7))
        cls.daily = pd.DataFrame({
            "session_date": dates,
            "snapshot_at": opens,
            "ctx_rms_day_return20": .012 + index / 100000,
            "ctx_example_signal": np.sin((index - 1) / 2),
            "ctx_second_signal": np.cos(index / 7),
            "future_1d_log_return": labels,
            "future_1d_available_at": ends,
            "future_3d_log_return": labels + labels.shift(-1) + labels.shift(-2),
            "future_3d_available_at": pd.Series(ends).shift(-2),
        })
        cls.freeze = pd.Timestamp("2026-08-01", tz="UTC")
        cls.scores, cls.bundles, cls.metadata = prequential_daily_forecasts(cls.daily)

    def test_fitting_uses_actual_label_end_and_excludes_equal_boundary(self):
        source = self.daily.copy()
        cutoff = pd.Timestamp("2026-07-01", tz="UTC")
        eligible = source.future_3d_available_at.lt(cutoff)
        last = source.index[eligible][-1]
        source.loc[last, "future_3d_available_at"] = cutoff
        model = fit_daily_model(source, cutoff, 3, min_train=10)
        expected = source.future_3d_available_at.lt(cutoff) & source.future_3d_log_return.notna()
        self.assertEqual(model.training_rows, int(expected.sum()))
        self.assertLess(model.last_label_available, cutoff)
        self.assertTrue(all(name.startswith("ctx_") for name in model.features))
        self.assertNotIn("future_3d_log_return", model.features)

    def test_future_outcomes_and_features_cannot_change_an_earlier_fit(self):
        cutoff = pd.Timestamp("2026-07-01", tz="UTC")
        original = fit_daily_model(self.daily, cutoff, 3, min_train=10)
        altered = self.daily.copy()
        future = ~altered.future_3d_available_at.lt(cutoff)
        altered.loc[future, "future_1d_log_return"] = 50.
        altered.loc[future, "future_3d_log_return"] = -50.
        altered.loc[future, "ctx_example_signal"] = 99999.
        refitted = fit_daily_model(altered, cutoff, 3, min_train=10)
        self.assertEqual(original.training_rows, refitted.training_rows)
        self.assertEqual(original.last_label_available, refitted.last_label_available)
        pd.testing.assert_frame_equal(
            original.predict(self.daily.tail(5)),
            refitted.predict(self.daily.tail(5)),
        )

    def test_august_and_september_use_the_same_frozen_daily_models(self):
        final = self.scores.loc[self.scores.session_date.ge("2026-08-01")]
        self.assertTrue(final.session_date.dt.month.isin([8, 9]).all())
        for horizon in (1, 3):
            self.assertTrue(final[f"daily_fit_cutoff_{horizon}d"].eq(self.freeze).all())
            self.assertEqual(final[f"daily_train_rows_{horizon}d"].nunique(), 1)
            self.assertTrue(final[f"daily_model_available_{horizon}d"].eq(1).all())
        for model in self.bundles.values():
            if model.last_label_available is not None:
                self.assertLess(model.last_label_available, model.cutoff)
            self.assertLessEqual(model.cutoff, self.freeze)
        changed = self.daily.copy()
        # Even labels that began in July but finished in August stay excluded.
        for horizon in (1, 3):
            unavailable = ~changed[f"future_{horizon}d_available_at"].lt(self.freeze)
            changed.loc[unavailable, f"future_{horizon}d_log_return"] = 999.
        repeated, _, _ = prequential_daily_forecasts(changed)
        pd.testing.assert_frame_equal(self.scores, repeated)

    def test_early_months_use_zero_return_and_past_only_direction_fallback(self):
        may = self.scores.loc[self.scores.session_date.dt.month.eq(5)]
        for horizon in (1, 3):
            self.assertTrue(may[f"daily_expected_log_return_{horizon}d"].eq(0).all())
            self.assertTrue(may[f"daily_expected_scaled_{horizon}d"].eq(0).all())
            self.assertTrue(may[f"daily_model_available_{horizon}d"].eq(0).all())
            self.assertTrue(may[f"daily_train_rows_{horizon}d"].eq(0).all())
            self.assertTrue(may[f"daily_p_down_{horizon}d"].eq(.5).all())
            self.assertTrue(self.scores[f"daily_p_down_{horizon}d"].between(0, 1).all())
        cutoff = pd.Timestamp("2026-06-01", tz="UTC")
        model = fit_daily_model(self.daily, cutoff, 1, min_train=40)
        self.assertIsNone(model.regressor)
        mature = self.daily.future_1d_available_at.lt(cutoff)
        self.assertAlmostEqual(
            model.base_p_down, self.daily.loc[mature, "future_1d_log_return"].lt(0).mean()
        )
        self.assertEqual(model.training_rows, mature.sum())

    def test_daily_forecast_join_waits_until_morning_assessment(self):
        date = self.daily.session_date.iloc[-1]
        local = pd.DatetimeIndex([
            date + pd.Timedelta(hours=7, minutes=59),
            date + pd.Timedelta(hours=8),
            date + pd.Timedelta(hours=9),
        ]).tz_localize(TZ)
        minute = pd.DataFrame({"decision_time": local.tz_convert("UTC")})
        joined = join_daily_forecasts(minute, self.scores)
        self.assertTrue(joined.loc[0, list(DAILY_SCORE_FEATURES)].isna().all())
        np.testing.assert_allclose(
            joined.loc[1, list(DAILY_SCORE_FEATURES)].to_numpy(dtype=float),
            joined.loc[2, list(DAILY_SCORE_FEATURES)].to_numpy(dtype=float),
        )

    def test_reprojection_weights_days_equally_despite_different_bar_counts(self):
        class ZeroBase:
            def predict(self, frame):
                return np.zeros(len(frame))

        dates = pd.bdate_range("2026-06-01", periods=10)
        index = np.arange(len(dates), dtype=float)
        calibration = pd.DataFrame({
            "decision_time": (dates + pd.Timedelta(hours=14)).tz_localize("UTC"),
            "daily_expected_scaled_1d": index / 10,
            "daily_expected_scaled_3d": np.sin(index),
            "target_scaled": .5 * index + np.cos(index),
        })
        duplicate = pd.concat([calibration, pd.concat(
            [calibration.iloc[[0]]] * 100, ignore_index=True
        )], ignore_index=True)
        base = ZeroBase()
        first = fit_reprojection(base, calibration)
        second = fit_reprojection(base, duplicate)
        self.assertIsNotNone(first.correction)
        np.testing.assert_allclose(first.predict(calibration), second.predict(calibration))
        short = fit_reprojection(base, calibration.head(4))
        self.assertIsNone(short.correction)
        np.testing.assert_allclose(short.predict(calibration), 0)


if __name__ == "__main__":
    unittest.main()
