"""Causal features and delayed, gap/roll-safe executable return labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK_SIZE = 0.001


def prepare_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    frame = raw.sort_values("ts_event").reset_index(drop=True).copy()
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    if frame.ts_event.duplicated().any():
        raise ValueError("Duplicate target timestamps; inspect the source before modelling")
    good = frame.data_condition.eq("available")
    breaks = (
        frame.ts_event.diff().ne(pd.Timedelta(minutes=1))
        | frame.instrument_id.ne(frame.instrument_id.shift())
        | ~good | ~good.shift(fill_value=False)
    )
    segment = breaks.cumsum()
    frame["segment"] = segment
    frame["decision_time"] = frame.ts_event + pd.Timedelta(minutes=1)
    features: dict[str, pd.Series] = {}

    def rolling(values, window, stat):
        return (values.groupby(segment, sort=False).rolling(window, min_periods=window)
                .agg(stat).reset_index(level=0, drop=True).sort_index())

    change = frame.close.groupby(segment).diff() / TICK_SIZE
    frame["sigma_1m_ticks"] = np.sqrt(rolling(change.pow(2), 60, "mean"))
    sigma = frame.sigma_1m_ticks.clip(lower=0.5)
    tr = pd.concat([(frame.high-frame.low)/TICK_SIZE,
                    (frame.high-frame.close.groupby(segment).shift()).abs()/TICK_SIZE,
                    (frame.low-frame.close.groupby(segment).shift()).abs()/TICK_SIZE], axis=1).max(axis=1)
    features["sigma_1m_ticks"] = frame.sigma_1m_ticks
    for lag in (1, 3, 5, 15, 30, 60):
        features[f"momentum_{lag}m_vol"] = frame.close.groupby(segment).diff(lag) / (TICK_SIZE*sigma*np.sqrt(lag))
    candle_range = (frame.high-frame.low).replace(0, np.nan)
    features["range_vol"] = (frame.high-frame.low)/(TICK_SIZE*sigma)
    features["body_vol"] = (frame.close-frame.open)/(TICK_SIZE*sigma)
    features["close_location"] = ((frame.close-frame.low)/candle_range).fillna(0.5)
    features["upper_wick_vol"] = (frame.high-frame[["open", "close"]].max(axis=1))/(TICK_SIZE*sigma)
    features["lower_wick_vol"] = (frame[["open", "close"]].min(axis=1)-frame.low)/(TICK_SIZE*sigma)
    features["log_volume"] = np.log1p(frame.volume)
    for window in (5, 15, 30, 60):
        mean = rolling(frame.close, window, "mean")
        std = rolling(frame.close, window, "std").replace(0, np.nan)
        features[f"close_z_{window}m"] = (frame.close-mean)/std
        features[f"range_position_{window}m"] = ((frame.close-rolling(frame.low, window, "min")) /
            (rolling(frame.high, window, "max")-rolling(frame.low, window, "min")).replace(0,np.nan))
        features[f"vol_ratio_{window}m"] = np.sqrt(rolling(change.pow(2),window,"mean"))/sigma
        features[f"atr_{window}m_vol"] = rolling(tr,window,"mean")/sigma
        features[f"volume_ratio_{window}m"] = frame.volume/rolling(frame.volume,window,"mean").replace(0,np.nan)
        features[f"efficiency_{window}m"] = change.groupby(segment).rolling(window,min_periods=window).sum().reset_index(level=0,drop=True).sort_index().abs()/rolling(change.abs(),window,"sum").replace(0,np.nan)
    eastern = frame.decision_time.dt.tz_convert("America/New_York")
    minute = eastern.dt.hour*60+eastern.dt.minute
    features["minute_sin"] = np.sin(2*np.pi*minute/1440)
    features["minute_cos"] = np.cos(2*np.pi*minute/1440)
    features["dow_sin"] = np.sin(2*np.pi*eastern.dt.dayofweek/7)
    features["dow_cos"] = np.cos(2*np.pi*eastern.dt.dayofweek/7)
    # Session clock uses local wall time, including across daylight-saving changes.
    session = (eastern.dt.tz_localize(None)-pd.Timedelta(hours=18)).dt.date
    keys = [session, frame.instrument_id]
    typical = (frame.high+frame.low+frame.close)/3
    vwap = (typical*frame.volume).groupby(keys).cumsum()/frame.volume.groupby(keys).cumsum().replace(0,np.nan)
    features["session_vwap_distance_vol"] = (frame.close-vwap)/(TICK_SIZE*sigma)

    # Auxiliary returns reset at their OWN contract switches, missing observations,
    # and gaps in the target stream. Never interpret a futures roll as a return.
    for market in ("wti", "lng_equity"):
        price = f"{market}_close"
        if price not in frame:
            continue
        observed = frame[f"{market}_available"].eq(1)
        aux_break = breaks | ~observed | ~observed.shift(fill_value=False) | frame[f"{market}_instrument_id"].ne(frame[f"{market}_instrument_id"].shift())
        aux_segment = aux_break.cumsum()
        log_price = np.log(frame[price])
        for lag in (1,5,15,30):
            features[f"{market}_return_{lag}m"] = log_price.groupby(aux_segment).diff(lag)
        for suffix in ("available", "age_minutes"):
            features[f"{market}_{suffix}"] = frame[f"{market}_{suffix}"]

    curve = ["ng_m1", "ng_m2", "ng_m3"]
    if all(f"{name}_close" in frame for name in curve):
        available = frame[[f"{name}_available" for name in curve]].eq(1).all(axis=1)
        available &= frame[[f"{name}_instrument_id" for name in curve]].nunique(axis=1).eq(3)
        changed = frame[[f"{name}_instrument_id" for name in curve]].ne(frame[[f"{name}_instrument_id" for name in curve]].shift()).any(axis=1)
        curve_segment = (breaks | changed | ~available | ~available.shift(fill_value=False)).cumsum()
        features["curve_available"] = available.astype("int8")
        for left,right in (("ng_m1","ng_m2"),("ng_m2","ng_m3")):
            spread = ((frame[f"{left}_close"]-frame[f"{right}_close"])/TICK_SIZE).where(available)
            name = f"{left}_{right}_spread_ticks"
            features[name] = spread
            for lag in (5,15,30):
                features[f"{name}_change_{lag}m_vol"] = spread.groupby(curve_segment).diff(lag)/sigma
        for name in curve:
            for suffix in ("available", "age_minutes"):
                features[f"{name}_{suffix}"] = frame[f"{name}_{suffix}"]
    # Existing join used bar START. Keep its storage value: at most one minute
    # conservative, never pull a later actual backwards into an earlier bar.
    # Historical storage actuals are deliberately excluded: the supplied file
    # mixes reconstructed original values and potentially revised history.
    for name in ("eia_minutes_since_release","eia_minutes_to_release","eia_post_release_60m","eia_release_window"):
        if name in frame:
            features[name] = frame[name]
    values = pd.DataFrame(features).replace([np.inf,-np.inf],np.nan).astype("float32")
    # All rolling windows are strictly historical; missing auxiliary values are
    # retained and imputed using the training partition only.
    for name in values:
        frame[name] = values[name]
    frame["features_ready"] = good & frame.sigma_1m_ticks.notna()
    audit = {"raw_rows":len(frame),"segments":int(segment.nunique()),
             "quality_excluded":int((~good).sum()),"feature_ready_rows":int(frame.features_ready.sum()),
             "features":len(values.columns),"first_bar":str(frame.ts_event.min()),"last_bar":str(frame.ts_event.max())}
    return frame, list(values.columns), audit


def executable_dataset(frame: pd.DataFrame, horizon: int, *, delay_bars: int=1,
                       stride: int=5, session_start: int=8, session_end: int=16) -> pd.DataFrame:
    """Completed bar t -> decision t+1 -> fill t+1+delay, then exit h later.

    Both fills use future bar OPEN prices, not the current close. Exact
    continuity through the exit bar is required. No stop/limit intrabar fills.
    """
    if horizon < 1 or delay_bars < 1 or stride < 1:
        raise ValueError("Horizon, full-bar delay and sampling stride must be positive")
    offset = 1+delay_bars
    end = offset+horizon
    result = frame.copy()
    result["entry_time"] = frame.ts_event.shift(-offset)
    result["exit_time"] = frame.ts_event.shift(-end)
    result["entry_price"] = frame.open.shift(-offset)
    result["exit_price"] = frame.open.shift(-end)
    continuous = frame.segment.eq(frame.segment.shift(-end))
    exact = result.exit_time.eq(frame.ts_event+pd.Timedelta(minutes=end))
    result["target_ticks"] = np.rint((result.exit_price-result.entry_price)/TICK_SIZE)
    result["scale_ticks"] = (frame.sigma_1m_ticks*np.sqrt(horizon)).clip(lower=2)
    result["target_scaled"] = result.target_ticks/result.scale_ticks
    entry_local = result.entry_time.dt.tz_convert("America/New_York")
    exit_local = result.exit_time.dt.tz_convert("America/New_York")
    regular = ((entry_local.dt.hour >= session_start) & (exit_local.dt.hour < session_end)
               & entry_local.dt.date.eq(exit_local.dt.date) & entry_local.dt.dayofweek.lt(5))
    # Event risk rule fixed before fitting: no position from 5m before through
    # 15m after scheduled EIA releases, including holidays in supplied calendar.
    event_ok = pd.Series(True,index=frame.index)
    if "eia_minutes_to_release" in frame:
        event_ok &= frame.eia_minutes_to_release.gt(end+5)
        event_ok &= frame.eia_minutes_since_release.gt(15)
    sampled = frame.decision_time.dt.minute.mod(stride).eq(0)
    valid = frame.features_ready & continuous & exact & regular & event_ok & sampled
    result = result.loc[valid].reset_index(drop=True)
    result["horizon_minutes"] = horizon
    return result
