"""Causal daily context for intraday natural-gas forecasts.

The research session is 08:00--16:00 America/New_York, not the exchange's
overnight trading day.  A day's historical snapshot is frozen at 08:00 and
contains only completed earlier sessions.  Today's realised OHLC and forward
labels are returned in ``daily_summary`` for diagnostics/training, NEVER as
``ctx_`` features.

Trend indicators use a roll-neutral *log return index*, not an absolute or
tradable back-adjusted price.  It accumulates valid open-to-close returns and
overnight returns only between consecutive valid sessions in the SAME contract.
At a roll or missing/invalid session, the overnight bridge is omitted.  Windows
count valid observed sessions; an invalid immediately preceding session leaves
the next snapshot missing rather than silently forwarding an older snapshot.
This convention preserves long lookbacks without inventing holiday returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/New_York"
MIN_BARS = 400


def _wilder_rsi(values: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI with the original arithmetic-mean initialisation."""
    change = values.diff().to_numpy(dtype=float)
    answer = np.full(len(values), np.nan)
    if len(values) <= period:
        return pd.Series(answer, index=values.index)
    gain = np.maximum(change, 0.0)
    loss = np.maximum(-change, 0.0)
    avg_gain = float(np.mean(gain[1:period + 1]))
    avg_loss = float(np.mean(loss[1:period + 1]))
    for i in range(period, len(values)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        answer[i] = (50.0 if avg_gain == avg_loss == 0 else
                     100.0 if avg_loss == 0 else
                     100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    return pd.Series(answer, index=values.index)


def _confirmed_divergence(price: pd.Series, rsi: pd.Series) -> pd.DataFrame:
    """Compare consecutive strict pivots, announcing only at right confirmation.

    A pivot on session j requires j-2,j-1,j+1,j+2.  Thus its first possible
    announcement is session j+2's close, entering features the following morning.
    The RSI attached to each pivot was available on the pivot's own close.
    """
    values = price.to_numpy(dtype=float)
    strength = rsi.to_numpy(dtype=float)
    bullish = np.zeros(len(values))
    bearish = np.zeros(len(values))
    last_low = last_high = None
    for confirmed in range(4, len(values)):
        pivot = confirmed - 2
        neighbours = np.r_[values[pivot - 2:pivot], values[pivot + 1:pivot + 3]]
        if not np.isfinite(strength[pivot]):
            continue
        if values[pivot] < neighbours.min():
            if last_low is not None:
                old_price, old_rsi = last_low
                bullish[confirmed] = float(values[pivot] < old_price and
                                           strength[pivot] > old_rsi)
            last_low = (values[pivot], strength[pivot])
        if values[pivot] > neighbours.max():
            if last_high is not None:
                old_price, old_rsi = last_high
                bearish[confirmed] = float(values[pivot] > old_price and
                                           strength[pivot] < old_rsi)
            last_high = (values[pivot], strength[pivot])
    return pd.DataFrame({"rsi_bull_div_confirmed": bullish,
                         "rsi_bear_div_confirmed": bearish}, index=price.index)


def _linked_index(daily: pd.DataFrame, prefix: str = "") -> None:
    """Attach causal log indices without treating contract switches as returns."""
    last_index = 0.0
    previous = None
    rows = []
    for row in daily.to_dict("records"):
        valid = bool(row[f"{prefix}valid"])
        item = {f"{prefix}log_index_{field}": np.nan
                for field in ("open", "high", "low", "close")}
        item[f"{prefix}roll_boundary"] = False
        item[f"{prefix}bridge_omitted"] = True
        if valid:
            previous_valid = previous is not None and previous[f"{prefix}valid"]
            same = previous_valid and (
                row[f"{prefix}instrument_id"] == previous[f"{prefix}instrument_id"])
            roll = previous_valid and not same
            bridge = (np.log(row[f"{prefix}open"] / previous[f"{prefix}close"])
                      if same else 0.0)
            index_open = last_index + bridge
            for field in ("open", "high", "low", "close"):
                item[f"{prefix}log_index_{field}"] = (
                    index_open + np.log(row[f"{prefix}{field}"] / row[f"{prefix}open"]))
            last_index = item[f"{prefix}log_index_close"]
            item[f"{prefix}roll_boundary"] = bool(roll)
            item[f"{prefix}bridge_omitted"] = not bool(same)
        rows.append(item)
        previous = row
    for name in rows[0]:
        daily[name] = [row[name] for row in rows]


def _summarise(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.ts_event.dt.tz_convert(TZ)
    dates = local.dt.tz_localize(None).dt.normalize()
    minutes = local.dt.hour * 60 + local.dt.minute
    regular = minutes.ge(480) & minutes.lt(960) & local.dt.dayofweek.lt(5)
    source = frame.loc[regular].copy()
    source["_day"] = dates.loc[regular]
    source["_minute"] = minutes.loc[regular]
    source["_good"] = (source.data_condition.eq("available")
                       if "data_condition" in source else True)
    source["_good"] &= source[["open", "high", "low", "close"]].gt(0).all(axis=1)
    groups = {day: group for day, group in source.groupby("_day", sort=True)}
    day_index = pd.date_range(dates.min(), dates.max(), freq="B")
    rows = []
    has_wti = all(name in frame for name in
                  ("wti_close", "wti_instrument_id", "wti_available", "wti_age_minutes"))
    for day in day_index:
        session_open = (day + pd.Timedelta(hours=8)).tz_localize(TZ).tz_convert("UTC")
        session_end = (day + pd.Timedelta(hours=16)).tz_localize(TZ).tz_convert("UTC")
        row = {"session_date": day, "snapshot_at": session_open,
               "session_open_time": session_open, "session_end_time": session_end,
               "available_at": session_end}
        group = groups.get(day, source.iloc[:0])
        good = group.loc[group["_good"]]
        for prefix, observed in (
            ("", good),
            ("wti_", group.loc[
                group.wti_available.eq(1) & group.wti_age_minutes.le(2)
                & group.wti_close.gt(0) & group.wti_instrument_id.notna()]
             if has_wti else group.iloc[:0]),
        ):
            count = len(observed)
            row[f"{prefix}bars"] = count
            row[f"{prefix}coverage"] = count / 480.0
            row[f"{prefix}valid"] = False
            row[f"{prefix}roll_inside_session"] = False
            for name in ("open", "high", "low", "close", "instrument_id"):
                row[f"{prefix}{name}"] = np.nan
            row[f"{prefix}open_time"] = pd.NaT
            row[f"{prefix}open_known_at"] = pd.NaT
            row[f"{prefix}end_time"] = pd.NaT
            if count:
                first, last = observed.iloc[0], observed.iloc[-1]
                instrument = f"{prefix}instrument_id"
                one_contract = observed[instrument].nunique() == 1
                row[f"{prefix}roll_inside_session"] = not one_contract
                row[f"{prefix}open"] = float(first["wti_close" if prefix else "open"])
                row[f"{prefix}close"] = float(last[f"{prefix}close"])
                row[f"{prefix}high"] = float(observed["wti_close" if prefix else "high"].max())
                row[f"{prefix}low"] = float(observed["wti_close" if prefix else "low"].min())
                row[f"{prefix}instrument_id"] = first[instrument]
                row[f"{prefix}open_time"] = first.ts_event
                row[f"{prefix}open_known_at"] = first.ts_event + pd.Timedelta(minutes=1)
                row[f"{prefix}end_time"] = last.ts_event + pd.Timedelta(minutes=1)
                row[f"{prefix}valid"] = bool(
                    count >= MIN_BARS and first["_minute"] <= 485
                    and last["_minute"] >= 955 and one_contract)
        rows.append(row)
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No weekday sessions in the supplied minute history")
    for name in ("open_time", "open_known_at", "end_time",
                 "wti_open_time", "wti_open_known_at", "wti_end_time"):
        daily[name] = pd.to_datetime(daily[name], utc=True)
    _linked_index(daily)
    _linked_index(daily, "wti_")
    daily["daily_log_return"] = np.log(daily.close / daily.open).where(daily.valid)
    daily["daily_range_log"] = np.log(daily.high / daily.low).where(daily.valid)
    daily["daily_return_ticks"] = ((daily.close - daily.open) / .001).where(daily.valid)
    daily["daily_range_ticks"] = ((daily.high - daily.low) / .001).where(daily.valid)
    daily["future_1d_log_return"] = daily.daily_log_return
    daily["future_1d_available_at"] = daily.available_at.where(daily.valid)
    valid3 = daily.valid & daily.valid.shift(-1, fill_value=False) & daily.valid.shift(-2, fill_value=False)
    daily["future_3d_log_return"] = (
        daily.log_index_close.shift(-2) - daily.log_index_open).where(valid3)
    daily["future_3d_available_at"] = daily.available_at.shift(-2).where(valid3)
    return daily


def _historical_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Features calculated at each completed close, before shifting to next day."""
    history = pd.DataFrame(index=daily.index)
    valid = daily.loc[daily.valid]
    close = valid.log_index_close
    # Exponentiation produces a dimensionless synthetic level, never raw futures.
    level = np.exp(close)
    for window in (3, 5, 10, 20, 60):
        history[f"return_{window}d"] = close.diff(window)
    for window in (5, 20, 60):
        mean = level.rolling(window, min_periods=window).mean()
        history[f"ma_distance_{window}d"] = level / mean - 1
        history[f"ma_slope_{window}d_5d"] = np.log(mean / mean.shift(5))
    mean20 = level.rolling(20, min_periods=20).mean()
    std20 = level.rolling(20, min_periods=20).std(ddof=1).replace(0, np.nan)
    history["bb20_z"] = (level - mean20) / std20
    history["bb20_position"] = (level - (mean20 - 2 * std20)) / (4 * std20)
    history["bb20_bandwidth"] = 4 * std20 / mean20
    history["bb20_upper_distance"] = level / (mean20 + 2 * std20) - 1
    history["bb20_lower_distance"] = level / (mean20 - 2 * std20) - 1
    # Internal frozen levels are used to map the live same-contract move onto
    # the same synthetic basis. They are excluded from returned model features.
    history["_index_close"] = close
    history["_index_level_ma20"] = mean20
    history["_index_level_std20"] = std20
    streak = 0
    streaks = []
    for price, mean in zip(level, mean20):
        if not np.isfinite(mean):
            streaks.append(np.nan)
            streak = 0
        else:
            streak = (max(streak, 0) + 1 if price > mean else
                      min(streak, 0) - 1 if price < mean else 0)
            streaks.append(streak)
    history["days_above_below_ma20"] = pd.Series(streaks, index=valid.index)
    rsi = _wilder_rsi(level)
    history["rsi14"] = rsi
    history["rsi14_change_5d"] = rsi.diff(5)
    divergence = _confirmed_divergence(level, rsi)
    for name in divergence:
        history[name] = divergence[name]
        history[name + "_recent5d"] = divergence[name].rolling(5, min_periods=1).max()
    # A clearly named slope disagreement proxy, separate from confirmed pivots.
    history["rsi_bull_slope_disagreement_5d"] = (
        close.diff(5).lt(0) & rsi.diff(5).gt(0)).astype(float).where(rsi.diff(5).notna())
    history["rsi_bear_slope_disagreement_5d"] = (
        close.diff(5).gt(0) & rsi.diff(5).lt(0)).astype(float).where(rsi.diff(5).notna())
    for name, series in (("day_return", valid.daily_log_return),
                         ("day_range", valid.daily_range_log)):
        history[f"prev_{name}"] = series
        for window in (5, 20):
            roll = series.rolling(window, min_periods=window)
            history[f"{name}_mean{window}"] = roll.mean()
            history[f"{name}_std{window}"] = roll.std(ddof=1)
            history[f"{name}_abs_mean{window}"] = series.abs().rolling(window, min_periods=window).mean()
    history["rms_day_return20"] = np.sqrt(
        valid.daily_log_return.pow(2).rolling(20, min_periods=20).mean())
    history["prev_day_return_ticks"] = valid.daily_return_ticks
    history["prev_day_range_ticks"] = valid.daily_range_ticks
    history["up_day_fraction20"] = valid.daily_log_return.gt(0).astype(float).rolling(20, min_periods=20).mean()
    history["prev_roll_boundary"] = daily.roll_boundary.astype(float).where(daily.valid)
    history["prev_bridge_omitted"] = daily.bridge_omitted.astype(float).where(daily.valid)
    history["invalid_sessions_last20"] = (~daily.valid).astype(float).rolling(20, min_periods=1).sum()
    history["prev_session_valid"] = daily.valid.astype(float)
    history["prev_session_coverage"] = daily.coverage
    history["prev_wti_session_valid"] = daily.wti_valid.astype(float)
    joint = daily.loc[daily.valid & daily.wti_valid]
    relative = joint.log_index_close - joint.wti_log_index_close
    history["ng_wti_relative_log_index"] = relative
    for window in (5, 20):
        history[f"ng_wti_relative_return_{window}d"] = relative.diff(window)
        history[f"wti_return_{window}d"] = joint.wti_log_index_close.diff(window)
    for window in (20, 60):
        rolling = relative.rolling(window, min_periods=window)
        history[f"ng_wti_relative_z{window}"] = (
            (relative - rolling.mean()) / rolling.std(ddof=1).replace(0, np.nan))
    history["prev_wti_roll_boundary"] = daily.wti_roll_boundary.astype(float).where(daily.wti_valid)
    return history


def add_daily_context(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict]:
    """Add completed-session context and causal current-session progress.

    ``frame`` is the output of advanced_data.prepare_features.  The returned
    daily table contains static ``ctx_*`` columns usable at ``snapshot_at`` and
    explicitly named future labels.  The enriched minute frame additionally
    contains current-session progress features using bars completed by
    ``decision_time``.  All returned model feature names begin with ``ctx_``.
    """
    required = {"ts_event", "decision_time", "open", "high", "low", "close", "instrument_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Daily context is missing input columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Daily context requires nonempty minute data")
    if not frame.ts_event.is_monotonic_increasing or frame.ts_event.duplicated().any():
        raise ValueError("Daily context requires unique, chronologically ordered bars")
    result = frame.copy()
    result["ts_event"] = pd.to_datetime(result.ts_event, utc=True)
    result["decision_time"] = pd.to_datetime(result.decision_time, utc=True)
    if (result.decision_time < result.ts_event + pd.Timedelta(minutes=1)).any():
        raise ValueError("A minute bar cannot be used before its completion")
    daily = _summarise(result)
    historical = _historical_features(daily)
    snapshot = historical.shift(1)
    static_names = []
    for name in historical:
        if not name.startswith("_"):
            feature = "ctx_" + name
            daily[feature] = snapshot[name]
            static_names.append(feature)
    daily["previous_session_end"] = daily.available_at.shift(1)
    daily["previous_session_valid"] = daily.valid.shift(1, fill_value=False)
    daily["daily_open_return_std20"] = daily.ctx_day_return_std20
    dayofyear = daily.session_date.dt.dayofyear
    for name, values in {
        "ctx_year_sin": np.sin(2 * np.pi * (dayofyear - 1) / 365.2425),
        "ctx_year_cos": np.cos(2 * np.pi * (dayofyear - 1) / 365.2425),
        "ctx_month_sin": np.sin(2 * np.pi * (daily.session_date.dt.month - 1) / 12),
        "ctx_month_cos": np.cos(2 * np.pi * (daily.session_date.dt.month - 1) / 12),
        "ctx_heating_season": daily.session_date.dt.month.isin([11, 12, 1, 2, 3]).astype(float),
        "ctx_cooling_season": daily.session_date.dt.month.isin([6, 7, 8]).astype(float),
    }.items():
        daily[name] = values
        static_names.append(name)
    # Selection uses wall-clock snapshots, never a backwards calendar-date join.
    # Before 08:00 today's calendar date has no snapshot; the prior snapshot is
    # still safe and its age is explicit.  Forward labels/OHLC are NOT joined.
    join_columns = ["snapshot_at", *static_names]
    joined = pd.merge_asof(
        result[["decision_time"]], daily[join_columns],
        left_on="decision_time", right_on="snapshot_at", direction="backward")
    for name in static_names:
        result[name] = joined[name].to_numpy()
    result["ctx_snapshot_age_hours"] = (
        (result.decision_time.reset_index(drop=True) - joined.snapshot_at)
        .dt.total_seconds().to_numpy() / 3600)
    feature_names = [*static_names, "ctx_snapshot_age_hours"]

    # Progress is computed independently of end-of-day validity.  In particular,
    # a later missing bar or roll cannot retroactively alter a morning feature.
    local = result.ts_event.dt.tz_convert(TZ)
    dates = local.dt.tz_localize(None).dt.normalize()
    minute = local.dt.hour * 60 + local.dt.minute
    in_session = minute.ge(480) & minute.lt(960) & local.dt.dayofweek.lt(5)
    good = (result.data_condition.eq("available")
            if "data_condition" in result else pd.Series(True, index=result.index))
    good &= result[["open", "high", "low", "close"]].gt(0).all(axis=1)
    active = in_session & good
    live = result.loc[active, ["open", "high", "low", "close", "instrument_id"]].copy()
    live["_day"] = dates.loc[active]
    live["_minute"] = minute.loc[active]
    byday = live.groupby("_day", sort=False)
    live_open = byday.open.transform("first")
    open_minute = byday["_minute"].transform("first")
    first_contract = byday.instrument_id.transform("first")
    # Once any observed contract switch occurs, this day's progress remains
    # invalid; switching back later must not revive a stitched return.
    contract_ok = live.instrument_id.eq(first_contract).groupby(live["_day"]).cummin()
    live_ok = open_minute.le(485) & contract_ok
    current_return = np.log(live.close / live_open).where(live_ok)
    current_range = np.log(byday.high.cummax() / byday.low.cummin()).where(live_ok)
    progress = pd.DataFrame(index=result.index)
    progress["ctx_today_open_close_log_return"] = current_return
    progress["ctx_today_range_log"] = current_range
    progress["ctx_today_return_vs_rms20"] = (
        progress.ctx_today_open_close_log_return / result.ctx_rms_day_return20.replace(0, np.nan))
    progress["ctx_today_range_vs_mean20"] = (
        progress.ctx_today_range_log / result.ctx_day_range_mean20.replace(0, np.nan))
    progress["ctx_today_return_vs_abs_mean20"] = (
        progress.ctx_today_open_close_log_return / result.ctx_day_return_abs_mean20.replace(0, np.nan))
    progress["ctx_session_elapsed_fraction"] = ((minute + 1 - 480) / 480).where(in_session)
    progress["ctx_today_roll_observed"] = (~contract_ok).astype(float)
    progress["ctx_today_observed_fraction"] = (
        (byday.cumcount() + 1) / (live["_minute"] + 1 - 480))
    progress["ctx_today_open_delay_minutes"] = open_minute - 480
    # Previous raw close and contract are historical; current session open is
    # derived solely from the running group, not the future session summary.
    prev_close = pd.Series(daily.close.shift(1).to_numpy(), index=daily.session_date)
    prev_id = pd.Series(daily.instrument_id.shift(1).to_numpy(), index=daily.session_date)
    prev_valid = pd.Series(daily.valid.shift(1, fill_value=False).to_numpy(), index=daily.session_date)
    previous_close_live = live["_day"].map(prev_close)
    same_contract = live.instrument_id.eq(live["_day"].map(prev_id))
    previous_ok = live["_day"].map(prev_valid).fillna(False).astype(bool)
    overnight = np.log(live_open / previous_close_live).where(same_contract & previous_ok, 0.0)
    progress["ctx_today_overnight_log_return"] = overnight.where(live_ok & previous_ok)
    progress["ctx_today_open_roll"] = (~same_contract).astype(float).where(previous_ok)
    prev_index = pd.Series(snapshot["_index_close"].to_numpy(), index=daily.session_date)
    mean20 = pd.Series(snapshot["_index_level_ma20"].to_numpy(), index=daily.session_date)
    std20 = pd.Series(snapshot["_index_level_std20"].to_numpy(), index=daily.session_date)
    live_level = np.exp(live["_day"].map(prev_index) + overnight + current_return)
    mean_live = live["_day"].map(mean20)
    std_live = live["_day"].map(std20).replace(0, np.nan)
    progress["ctx_live_bb20_z"] = (live_level - mean_live) / std_live
    progress["ctx_live_bb20_position"] = (live_level - mean_live + 2 * std_live) / (4 * std_live)
    progress["ctx_live_ma20_distance"] = live_level / mean_live - 1
    progress["ctx_remaining_reversion_to_ma20"] = -progress.ctx_live_ma20_distance
    for name in progress:
        result[name] = progress[name]
        feature_names.append(name)
    # Keep current-day realised columns and labels out of feature_names.
    result[feature_names] = result[feature_names].replace(
        [np.inf, -np.inf], np.nan).astype("float32")
    daily[static_names] = daily[static_names].replace([np.inf, -np.inf], np.nan)
    audit = {
        "session_definition": "08:00-16:00 America/New_York; bar start in [08:00,16:00)",
        "minimum_good_bars": MIN_BARS,
        "endpoint_tolerance_minutes": 5,
        "sessions": len(daily),
        "valid_sessions": int(daily.valid.sum()),
        "invalid_sessions": int((~daily.valid).sum()),
        "valid_wti_sessions": int(daily.wti_valid.sum()),
        "ng_roll_boundaries": int(daily.roll_boundary.sum()),
        "wti_roll_boundaries": int(daily.wti_roll_boundary.sum()),
        "future_1d_labels": int(daily.future_1d_log_return.notna().sum()),
        "future_3d_labels": int(daily.future_3d_log_return.notna().sum()),
        "snapshots_with_20d_bollinger": int(daily.ctx_bb20_z.notna().sum()),
        "snapshots_with_60d_return": int(daily.ctx_return_60d.notna().sum()),
        "context_features": len(feature_names),
        "index_convention": (
            "Dimensionless roll-neutral log-return index: valid session open-close "
            "plus overnight bridge only across consecutive valid same-contract sessions; "
            "omit bridge at rolls/gaps. Indicator windows count valid observed sessions."),
        "wti_convention": (
            "Sampled minute closes at 08:00-16:00, freshness <=2m and >=400 observations; "
            "relative index is NG minus WTI log-return index, not a fair-value spread."),
        "causality": (
            "Static snapshot uses previous completed sessions only; invalid previous "
            "session is not forward-filled. Current-session features use cumulative "
            "completed bars independently of eventual session validity."),
        "rsi_divergence": "Strict two-left/two-right price pivots, published on second right close.",
        "seasonality": (
            "Calendar phase and demand-season flags only; calendar terms use no "
            "fitted climatology. Multi-year repetition improves coverage but does "
            "not by itself establish a stable seasonal trading edge."
        ),
    }
    return result, feature_names, daily, audit
