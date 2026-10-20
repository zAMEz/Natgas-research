"""Offline morning context and short-horizon forecasts from a frozen bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from advanced_data import prepare_features
from daily_context import add_daily_context
from daily_models import join_daily_forecasts
from train_context import DEFAULT_DATA


def forecast_as_of(bundle: dict,raw: pd.DataFrame,as_of: str | pd.Timestamp,rows: int=1):
    cutoff=pd.Timestamp(as_of)
    cutoff=cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    policy_cutoff=pd.Timestamp(bundle["policy_selected_through"])
    if cutoff<=policy_cutoff:
        raise ValueError("Use a time after this bundle's policy cutoff; earlier predictions would be in-sample")
    times=pd.to_datetime(raw.ts_event,utc=True)
    completed=raw.loc[(times+pd.Timedelta(minutes=1)).le(cutoff)].copy()
    frame,_,_=prepare_features(completed)
    frame,_,daily,_=add_daily_context(frame)
    scored=[bundle["daily_models"][h].predict(daily) for h in (1,3)]
    scores=scored[0].merge(scored[1],on="session_date",validate="one_to_one")
    frame=join_daily_forecasts(frame,scores)
    config=bundle["config"];horizon=bundle["horizon"]
    use=frame.features_ready & frame.decision_time.dt.minute.mod(config["sample_stride"]).eq(0) & frame.decision_time.gt(policy_cutoff)
    selected=frame.loc[use].tail(rows).copy()
    if selected.empty: raise ValueError("No completed, feature-ready decisions after the policy cutoff")
    scale=(selected.sigma_1m_ticks*np.sqrt(horizon)).clip(lower=2).to_numpy()
    entry=selected.decision_time+pd.Timedelta(minutes=config["delay_bars"])
    exit_time=entry+pd.Timedelta(minutes=horizon)
    entry_local=entry.dt.tz_convert("America/New_York")
    exit_local=exit_time.dt.tz_convert("America/New_York")
    session_ok=(entry_local.dt.hour.ge(8)&exit_local.dt.hour.lt(16)&entry_local.dt.dayofweek.lt(5)&entry_local.dt.date.eq(exit_local.dt.date)).to_numpy()
    event_ok=(selected.eia_minutes_to_release.gt(1+config["delay_bars"]+horizon+5)&selected.eia_minutes_since_release.gt(15)).to_numpy()
    output=[]
    for name,model in bundle["models"].items():
        pred=model.predict(selected)*scale
        policy=bundle["policies"][name]
        allowed=policy["enabled"] & (np.abs(pred)>config["cost_ticks"]+policy["edge"]) & session_ok & event_ok
        output.append(pd.DataFrame({"decision_time":selected.decision_time.to_numpy(),"model":name,
            "horizon_minutes":horizon,"predicted_ticks":pred,"policy_enabled":policy["enabled"],
            "policy_screen_direction":np.where(allowed,np.sign(pred),0).astype(int),
            "within_session":session_ok,"clear_of_eia":event_ok,"deployment_enabled":False,
            "hypothetical_entry":entry.to_numpy(),"hypothetical_exit":exit_time.to_numpy(),
            "daily_p_down_1d":selected.daily_p_down_1d.to_numpy(),"daily_p_down_3d":selected.daily_p_down_3d.to_numpy()}))
    # The assessment has only information known at its snapshot, plus frozen
    # model forecasts. No current close, final range, or future labels exported.
    historical=["session_date","snapshot_at","previous_session_end"]+[c for c in daily if c.startswith("ctx_")]
    assessment=daily.loc[daily.snapshot_at.le(cutoff),historical].merge(scores,on="session_date").tail(1)
    return pd.concat(output,ignore_index=True).sort_values(["decision_time","model"]),assessment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True,help="Trusted local joblib bundle from train_context.py")
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--as-of",required=True)
    parser.add_argument("--rows",type=int,default=1)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--assessment-output",type=Path)
    args=parser.parse_args()
    if args.rows<1: parser.error("rows must be positive")
    for path in (args.output,args.assessment_output):
        if path is not None and path.exists(): parser.error(f"Output exists; choose a new filename: {path}")
    forecasts,assessment=forecast_as_of(joblib.load(args.bundle),pd.read_parquet(args.data),args.as_of,args.rows)
    for path,table in ((args.output,forecasts),(args.assessment_output,assessment)):
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x") as stream: table.to_csv(stream,index=False)
    print(f"Saved {len(forecasts)} offline forecasts. Deployment disabled; no orders or position accounting.")


if __name__=="__main__":
    main()
