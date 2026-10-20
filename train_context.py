"""Test daily context, daily forecasts and learned daily-to-intraday adjustment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,r2_score,brier_score_loss

from advanced_data import prepare_features,executable_dataset
from advanced_estimators import fit_predict_models,FittedRegressor
from advanced_backtest import simulate,session_days,daily_pnl,metrics,select_policy,block_bootstrap,reality_check
from train_advanced import Config,DEFAULT_DATA,PROJECT,log,json_write,file_hash,split_windows,forecast_metrics,summarize,select_research_candidate
from daily_context import add_daily_context
from daily_models import (DAILY_SCORE_FEATURES,prequential_daily_forecasts,join_daily_forecasts,
                          fit_reprojection,DailyDriftRegressor)

CANDIDATES=("intraday_lightgbm","context_lightgbm","context_ebm","context_random_forest",
            "context_extra_trees","context_ridge","context_ensemble","stacked_lightgbm",
            "reprojected_lightgbm","daily_drift_only")


def fit_lgbm(windows,features,config):
    """Same configuration as the registered intraday-only LightGBM benchmark."""
    train,stop=windows["train"],windows["stop"]
    imputer=SimpleImputer(strategy="median",keep_empty_features=True)
    x=imputer.fit_transform(train[features].to_numpy(dtype=float))
    v=imputer.transform(stop[features].to_numpy(dtype=float))
    estimator=LGBMRegressor(objective="regression",n_estimators=100 if config.quick else 600,
        learning_rate=.025,num_leaves=15,max_depth=5,min_child_samples=200,min_split_gain=.01,
        reg_alpha=1.,reg_lambda=10.,colsample_bytree=.8,subsample=.8,subsample_freq=1,max_bin=63,
        deterministic=True,force_col_wise=True,random_state=config.seed,n_jobs=config.threads,
        verbosity=-1,importance_type="gain")
    estimator.fit(pd.DataFrame(x,columns=features),np.clip(train.target_scaled,-8,8),
        eval_set=[(pd.DataFrame(v,columns=features),stop.target_scaled.to_numpy())],eval_metric="l2",
        callbacks=[early_stopping(20 if config.quick else 40,first_metric_only=True,verbose=False),log_evaluation(0)])
    return FittedRegressor(tuple(features),imputer,estimator)


def run_context_fold(data,base_features,context_features,start,end,phase,horizon,config,output,daily_bundles):
    fold=start.strftime("%Y-%m")
    windows=split_windows(data,start,end,config.train_days)
    log(f"{phase} {fold} {horizon}m: "+", ".join(f"{k}={len(v)}" for k,v in windows.items()))
    combined=base_features+context_features
    sets={key:value[combined] for key,value in windows.items() if key!="train"}
    suite,_,training=fit_predict_models(windows["train"][combined],windows["train"].target_scaled.to_numpy(),
        windows["stop"][combined],windows["stop"].target_scaled.to_numpy(),sets,
        seed=config.seed,threads=config.threads,quick=config.quick)
    models={f"context_{name}":model for name,model in suite.items() if name not in ("lightgbm_q10","lightgbm_q90")}
    models["intraday_lightgbm"]=fit_lgbm(windows,base_features,config)
    models["stacked_lightgbm"]=fit_lgbm(windows,combined+list(DAILY_SCORE_FEATURES),config)
    models["reprojected_lightgbm"]=fit_reprojection(models["intraday_lightgbm"],windows["calibration"])
    models["daily_drift_only"]=DailyDriftRegressor(horizon)
    test=windows["test"]; validation=windows["validation"]; days=session_days(test)
    forecasts=[];strategies=[];policies=[];trials=[];trades_all=[];daily_all=[];predictions=[];paired=[]
    policy_map={}
    base_prediction=models["intraday_lightgbm"].predict(test)*test.scale_ticks.to_numpy()
    base_error=np.abs(test.target_ticks.to_numpy()-base_prediction)
    for name in CANDIDATES:
        model=models[name]
        pv=model.predict(validation)*validation.scale_ticks.to_numpy()
        policy,candidates=select_policy(validation,pv,config.cost_ticks)
        prefix={"phase":phase,"fold":fold,"horizon":horizon,"model":name}
        policy_map[name]=policy
        policies.append({**prefix,**policy})
        trials.extend({**prefix,**candidate} for candidate in candidates)
        prediction=model.predict(test)*test.scale_ticks.to_numpy()
        forecasts.append({**prefix,**forecast_metrics(test,prediction)})
        trades=simulate(test,prediction,config.cost_ticks,policy["edge"])
        for cost in sorted(set(config.stress_costs+(config.cost_ticks,))):
            adjusted=trades.copy(); adjusted["cost_ticks"]=cost;adjusted["net_ticks"]=adjusted.gross_ticks-cost
            strategies.append({**prefix,"cost_ticks":cost,**metrics(adjusted,days)})
        if len(trades): trades_all.append(trades.assign(**prefix))
        daily_all.append(pd.DataFrame({"date":days,"net_ticks":daily_pnl(trades,days).to_numpy(),**prefix}))
        predictions.append(test[["decision_time","entry_time","exit_time","target_ticks","scale_ticks"]].assign(**prefix,predicted_ticks=prediction))
        improvement=base_error-np.abs(test.target_ticks.to_numpy()-prediction)
        date=test.decision_time.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        paired.append(pd.DataFrame({"date":date,"mae_improvement_ticks":improvement}).groupby("date",as_index=False).mean().assign(**prefix))
    forecasts.append({"phase":phase,"fold":fold,"horizon":horizon,"model":"zero_change",**forecast_metrics(test,np.zeros(len(test)))})
    folder=output/f"{phase}_{fold}_{horizon}m";folder.mkdir()
    for filename,rows in (("forecast_metrics",forecasts),("strategy_metrics",strategies),("policies",policies),
                          ("validation_trials",trials),("training_metadata",training)):
        pd.DataFrame(rows).to_csv(folder/f"{filename}.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(folder/"predictions.parquet",index=False)
    dayframe=pd.concat(daily_all,ignore_index=True)
    trades=pd.concat(trades_all,ignore_index=True) if trades_all else pd.DataFrame(columns=["phase","fold","horizon","model","entry_time","exit_time","net_ticks","gross_ticks","cost_ticks"])
    trades.to_csv(folder/"trades.csv",index=False)
    dayframe.to_csv(folder/"daily_pnl.csv",index=False)
    if phase=="holdout":
        latest_daily={h:model for (cutoff,h),model in daily_bundles.items() if pd.Timestamp(cutoff)==start}
        joblib.dump({"models":models,"daily_models":latest_daily,"policies":policy_map,
            "base_features":base_features,"context_features":context_features,
            "daily_score_features":list(DAILY_SCORE_FEATURES),"config":asdict(config),"horizon":horizon,
            "trained_labels_through":str(windows["train"].exit_time.max()),
            "policy_selected_through":str(validation.exit_time.max()),
            "deployment_enabled":False,"purpose":"daily context retrospective experiment"},folder/"models.joblib",compress=3)
        ebm=models["context_ebm"].estimator
        pd.DataFrame({"term":ebm.term_names_,"magnitude":ebm.term_importances()}).sort_values("magnitude",ascending=False).to_csv(folder/"ebm_terms.csv",index=False)
    log(f"{phase} {fold} {horizon}m policies: "+", ".join(p['model'] for p in policies if p['enabled'])+" (others disabled)")
    return {"forecasts":forecasts,"strategies":strategies,"policies":policies,"daily":dayframe,"trades":trades,"paired":pd.concat(paired,ignore_index=True)}


def daily_diagnostics(daily,scores,development_start,holdout_start):
    joint=daily.merge(scores,on="session_date",validate="one_to_one")
    joint["phase"]=np.where(joint.snapshot_at.ge(holdout_start),"holdout",np.where(joint.snapshot_at.ge(development_start),"development","earlier"))
    metrics_rows=[];hypothesis=[]
    for phase,group in joint.loc[joint.phase.ne("earlier")].groupby("phase"):
        for horizon in (1,3):
            # Labels that finish beyond the development/holdout boundary do not
            # enter development diagnostics, just like training label purging.
            upper=holdout_start if phase=="development" else pd.Timestamp.max.tz_localize("UTC")
            use=group.loc[group[f"future_{horizon}d_log_return"].notna()&group[f"future_{horizon}d_available_at"].lt(upper)&group[f"daily_model_available_{horizon}d"].eq(1)]
            if use.empty: continue
            y=use[f"future_{horizon}d_log_return"].to_numpy()
            pred=use[f"daily_expected_log_return_{horizon}d"].to_numpy()
            down=(y<0).astype(int)
            metrics_rows.append({"phase":phase,"horizon_days":horizon,"sessions":len(use),
                "mae_return_pct":mean_absolute_error(y,pred)*100,"zero_mae_return_pct":np.abs(y).mean()*100,
                "r2":r2_score(y,pred),"direction_accuracy":float((np.sign(y)==np.sign(pred)).mean()),
                "brier_down":brier_score_loss(down,use[f"daily_p_down_{horizon}d"]),
                "base_rate_brier":brier_score_loss(down,use[f"daily_base_p_down_{horizon}d"])})
            states={"all":pd.Series(True,index=use.index),
                "high_bb_z_above_1_and_3days_above_ma":use.ctx_bb20_z.gt(1)&use.ctx_days_above_below_ma20.ge(3),
                "low_bb_z_below_minus1_and_3days_below_ma":use.ctx_bb20_z.lt(-1)&use.ctx_days_above_below_ma20.le(-3),
                "rsi_above_70":use.ctx_rsi14.gt(70),"rsi_below_30":use.ctx_rsi14.lt(30)}
            for regime,mask in states.items():
                target=use.loc[mask,f"future_{horizon}d_log_return"]
                hypothesis.append({"phase":phase,"horizon_days":horizon,"regime":regime,"sessions":len(target),
                    "fraction_down":float(target.lt(0).mean()) if len(target) else np.nan,
                    "mean_forward_log_return_pct":float(target.mean()*100) if len(target) else np.nan})
    return joint,pd.DataFrame(metrics_rows),pd.DataFrame(hypothesis)


def render_report(output,config,summary,forecasts,selection,audit,daily_metrics,hypothesis,paired_summary):
    history_years=(pd.Timestamp(audit["last_bar"])-pd.Timestamp(audit["first_bar"])).days/365.25
    hold=forecasts.loc[forecasts.phase.eq("holdout")]
    pivot=hold.pivot(index="model",columns="horizon",values="mae_ticks")
    final=summary.loc[summary.phase.eq("holdout")&summary.cost_ticks.eq(config.cost_ticks)]
    active=summary.loc[summary.trades.gt(0)&summary.cost_ticks.eq(config.cost_ticks)]
    lines=["# Daily context experiment", "",f"Frozen selection before final evaluation: **{selection['model']}**, horizon **{selection['horizon']}**.","",
        "## What changed","",
        f"The same {audit['base_features']} intraday inputs are compared with {audit['daily_context']['context_features']} daily/context features on identical executable rows. The daily history has {audit['daily_context']['valid_sessions']} valid sessions.",
        "Research day = 08:00–16:00 New York, matching the trading window. At 08:00 the historical assessment uses only prior completed sessions. Intraday progress updates after each completed minute.",
        "Daily inputs include open-to-close and high-low swings, 3/5/10/20/60-session trends, sustained highs/lows, Bollinger20 bands, Wilder RSI14, confirmed two-right-session RSI divergence, calendar seasonality and a normalized NG–WTI relative return index. Raw prices across futures rolls are not treated as trend changes.",
        "Separate ridge/logistic models forecast one- and three-session moves once per day. Their historical scores come from earlier monthly fits; final-period daily model parameters are frozen at the configured holdout start. Three-session labels must have finished before training.",
        "`stacked_lightgbm` learns from both context and those prior-generated daily forecasts. `reprojected_lightgbm` adjusts the intraday forecast using a calibration fit with one residual observation per session. `daily_drift_only` tests uniform allocation of the daily expected move across 480 minutes, explicitly a modelling assumption.",
        "", "## Final-period mean absolute error (ticks)","",pivot.round(4).to_markdown(),
        "", "## Paired change versus intraday-only LightGBM", "",
        "Positive improvement means lower error. Intervals resample five-session blocks of daily mean paired errors; they are descriptive and do not correct all past/adaptive experiments.","",
        paired_summary.loc[paired_summary.phase.eq("holdout")].round(5).to_markdown(index=False),
        "", "## Strategies that actually traded", "",
        (active[["phase","horizon","model","trades","net_ticks","net_usd_per_ng_contract","profit_factor","max_drawdown_ticks"]].round(4).to_markdown(index=False) if len(active) else "None passed the preceding validation trading gate."),
        "", "## Final policy results", "",
        final[["horizon","model","trades","net_ticks","mean_daily_ci_low","mean_daily_ci_high"]].round(4).to_markdown(index=False),
        "", "## Daily forecast diagnostics (one observation per session)", "",daily_metrics.round(5).to_markdown(index=False),
        "", "## Does an extended high precede a fall?", "",
        "These are retrospective conditional frequencies, not probabilities guaranteed for future trading. Three-session targets overlap and small regime samples are especially unreliable.","",
        hypothesis.round(4).to_markdown(index=False),
        "", "## Limits and reproducibility", "",
        f"The source spans about {history_years:.1f} years and {audit['daily_context']['valid_sessions']} valid daily sessions. That provides repeated calendar cycles, but calendar terms still cannot establish a durable seasonal trading edge without forward confirmation. Intraday rows are not independent copies of each daily forecast.",
        "The NG/WTI feature compares normalized roll-neutral return indices, not a raw price subtraction or a fixed fair-value relationship. WTI uses fresh sampled minute closes, not verified daily settlement prices.",
        "The three-session daily target is a research return index with roll bridges omitted; it is not a simulated three-day futures P&L. Actual trading P&L below uses delayed opens of the same contract and never crosses a roll.",
        "Repeated inspection of an evaluation period makes later runs exploratory, even when model selection precedes evaluation. All deployment flags remain false.",
        f"The execution rules use a {config.cost_ticks:g}-tick round-trip selection cost. The 1/2/3/4-tick stress results in strategy_summary.csv preserve exactly the same trades selected under that scenario. Missing future price paths are excluded retrospectively. Zero-trade days are included within eligible sessions.",
        "Model explanations are not p-values. The retrained intraday-only baseline is the primary ablation for incremental daily context; forecast improvement alone does not establish after-cost profit.",
        "", "Research rationale: [LONG_TERM_CONTEXT_RESEARCH.md](../LONG_TERM_CONTEXT_RESEARCH.md).",""]
    (output/"context_report.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--output",type=Path,default=PROJECT/"model_output_context")
    parser.add_argument("--horizons",nargs="+",type=int,default=[5,15,30])
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--cost-ticks",type=float,default=2.)
    parser.add_argument("--development-start",default="2026-05-01")
    parser.add_argument("--holdout-start",default="2026-08-01")
    parser.add_argument("--quick",action="store_true")
    args=parser.parse_args()
    if args.output.exists(): parser.error("Choose a new output directory; prior experiments are preserved")
    config=Config(horizons=tuple(args.horizons),cost_ticks=args.cost_ticks,threads=args.threads,quick=args.quick)
    development=pd.Timestamp(args.development_start,tz="UTC");holdout=pd.Timestamp(args.holdout_start,tz="UTC")
    if development>=holdout or args.cost_ticks<0 or args.threads<1 or any(h<1 for h in config.horizons): parser.error("Invalid dates, horizons, threads or costs")
    args.output.mkdir(parents=True)
    sources=("train_context.py","daily_context.py","daily_models.py","advanced_data.py","advanced_estimators.py","advanced_backtest.py","train_advanced.py")
    json_write(args.output/"manifest.json",{"config":asdict(config),"created_utc":datetime.now(timezone.utc).isoformat(),
        "data":str(args.data.resolve()),"data_sha256":file_hash(args.data),"source_sha256":{p:file_hash(PROJECT/p) for p in sources},
        "development_start":str(development),"holdout_start":str(holdout),"candidate_models":CANDIDATES,
        "daily_models":{"ridge_alpha":100.,"logistic_C":.05,"minimum_completed_labels":40,"refit":"monthly; frozen at final start"},
        "historical_dates_previously_inspected":True,"deployment_enabled":False})
    raw=pd.read_parquet(args.data)
    frame,base_features,audit=prepare_features(raw)
    frame,context_features,daily,context_audit=add_daily_context(frame)
    scores,bundles,metadata=prequential_daily_forecasts(daily,holdout_start=holdout)
    frame=join_daily_forecasts(frame,scores)
    audit["base_features"]=len(base_features);audit["daily_context"]=context_audit
    json_write(args.output/"data_audit.json",audit)
    json_write(args.output/"feature_list.json",{"intraday":base_features,"context":context_features,"daily_scores":DAILY_SCORE_FEATURES})
    daily.to_parquet(args.output/"daily_summary.parquet",index=False)
    daily.to_csv(args.output/"daily_summary.csv",index=False)
    scores.to_csv(args.output/"daily_forecasts.csv",index=False)
    pd.DataFrame(metadata).to_csv(args.output/"daily_fit_audit.csv",index=False)
    joblib.dump(bundles,args.output/"daily_models.joblib",compress=3)
    datasets={h:executable_dataset(frame,h) for h in config.horizons}
    log(f"Daily layer: {context_audit['valid_sessions']} valid sessions, {len(context_features)} context features; {len(base_features)} intraday features")
    results=[]
    for start in pd.date_range(development,holdout,freq="MS",inclusive="left"):
        for horizon,data in datasets.items():
            results.append(run_context_fold(data,base_features,context_features,start,start+pd.DateOffset(months=1),"development",horizon,config,args.output,bundles))
    preliminary,_,_=summarize(results,config)
    selection=select_research_candidate(preliminary,config)
    json_write(args.output/"frozen_selection.json",{**selection,"frozen_utc":datetime.now(timezone.utc).isoformat()})
    log(f"Frozen selection: {selection}")
    end=frame.ts_event.max()+pd.Timedelta(minutes=1)
    for horizon,data in datasets.items():
        results.append(run_context_fold(data,base_features,context_features,holdout,end,"holdout",horizon,config,args.output,bundles))
    summary,day_pnl,trades=summarize(results,config)
    summary.to_csv(args.output/"strategy_summary.csv",index=False)
    day_pnl.to_csv(args.output/"daily_pnl.csv",index=False)
    trades.to_csv(args.output/"all_trades.csv",index=False)
    forecasts=pd.DataFrame([row for result in results for row in result['forecasts']])
    forecasts.to_csv(args.output/"forecast_by_fold.csv",index=False)
    means=forecasts.groupby(["phase","horizon","model"],as_index=False)[["mae_ticks","rmse_ticks","r2","correlation","direction_accuracy_nonzero"]].mean()
    means.to_csv(args.output/"forecast_summary.csv",index=False)
    paired=pd.concat([result['paired'] for result in results],ignore_index=True)
    paired.to_csv(args.output/"paired_daily_forecast_errors.csv",index=False)
    paired_rows=[]
    for (phase,horizon,model),group in paired.groupby(["phase","horizon","model"]):
        ci=block_bootstrap(group.mae_improvement_ticks.to_numpy())
        paired_rows.append({"phase":phase,"horizon":horizon,"model":model,
            "mean_daily_mae_improvement":group.mae_improvement_ticks.mean(),
            "ci_low":ci['mean_daily_ci_low'],"ci_high":ci['mean_daily_ci_high']})
    paired_summary=pd.DataFrame(paired_rows);paired_summary.to_csv(args.output/"paired_forecast_comparison.csv",index=False)
    joint,daily_metrics,hypothesis=daily_diagnostics(daily,scores,development,holdout)
    joint.to_csv(args.output/"daily_assessment.csv",index=False)
    daily_metrics.to_csv(args.output/"daily_forecast_metrics.csv",index=False)
    hypothesis.to_csv(args.output/"mean_reversion_hypothesis.csv",index=False)
    checks={}
    for phase,group in day_pnl.groupby("phase"):
        matrix=group.pivot_table(index="date",columns=["horizon","model"],values="net_ticks",aggfunc="sum").fillna(0)
        checks[phase]=reality_check(matrix.to_numpy())
    json_write(args.output/"family_tests.json",checks)
    render_report(args.output,config,summary,means,selection,audit,daily_metrics,hypothesis,paired_summary)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    valid=daily.loc[daily.valid]
    axes[0].plot(valid.session_date,valid.daily_log_return*100,label="Open to close (%)")
    axes[0].plot(valid.session_date,valid.daily_range_log*100,alpha=.45,label="High-low log range (%)")
    axes[0].set_title("Observed daily swings");axes[0].legend(fontsize=8)
    axes[1].plot(daily.session_date,daily.ctx_bb20_z,label="Prior-session Bollinger z")
    axes[1].axhline(2,color="gray",ls="--");axes[1].axhline(-2,color="gray",ls="--")
    axes[1].set_title("Context available at the session open");axes[1].legend(fontsize=8)
    for ax in axes: ax.tick_params(axis="x",rotation=30)
    fig.tight_layout();fig.savefig(args.output/"daily_context.png",dpi=160);plt.close(fig)
    log(f"Completed: {args.output/'context_report.md'}")


if __name__=="__main__":
    main()
