"""Registered chronological NG experiment: models, meta-filter, costs and audit.

This is a research backtest. Reusing previously inspected history cannot establish
an independent live edge. The holdout below is withheld from THIS experiment's
selection, but was present in the older research. No brokerage integration exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from advanced_data import prepare_features, executable_dataset
from advanced_estimators import fit_predict_models
from advanced_backtest import (simulate, session_days, daily_pnl, metrics,
                               select_policy, block_bootstrap, reality_check)

PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT/"data/ng_augmented_ohlcv_1m_2025-09-24_2026-09-24_model.parquet"
MODEL_NAMES = ("lightgbm","ebm","random_forest","extra_trees","ridge","ensemble","lightgbm_meta")


@dataclass(frozen=True)
class Config:
    horizons: tuple[int,...] = (5,15,30)
    cost_ticks: float = 2.0
    stress_costs: tuple[float,...] = (1.0,2.0,3.0,4.0)
    delay_bars: int = 1
    sample_stride: int = 5
    train_days: int = 180
    threads: int = 4
    seed: int = 42
    quick: bool = False


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}",flush=True)


def json_write(path: Path, value):
    path.write_text(json.dumps(value,indent=2,default=str)+"\n")


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def split_windows(data: pd.DataFrame,test_start: pd.Timestamp,test_end: pd.Timestamp,
                  train_days: int=180) -> dict[str,pd.DataFrame]:
    cal_start = test_start-pd.DateOffset(months=1)
    valid_start = cal_start+pd.Timedelta(days=14)
    stop_start = cal_start-pd.Timedelta(days=14)
    train_start = stop_start-pd.Timedelta(days=train_days)
    bounds={"train":(train_start,stop_start),"stop":(stop_start,cal_start),
            "calibration":(cal_start,valid_start),"validation":(valid_start,test_start),
            "test":(test_start,test_end)}
    result={}
    for name,(start,end) in bounds.items():
        # Purge by actual exit; a 30m target has a longer purge than a 5m target.
        result[name]=data.loc[data.decision_time.ge(start)&data.exit_time.lt(end)].copy().reset_index(drop=True)
        if len(result[name]) < (300 if name=="train" else 30):
            raise ValueError(f"Too few rows in {name}: {len(result[name])}")
    return result


def meta_features(x: pd.DataFrame, prediction: np.ndarray, scale: np.ndarray,cost: float) -> pd.DataFrame:
    result=x.copy()
    result["base_prediction_scaled"]=prediction
    result["base_abs_prediction_scaled"]=np.abs(prediction)
    result["base_direction"]=np.sign(prediction)
    result["cost_scaled"]=cost/scale
    return result


def fit_meta(windows,features,predictions,config):
    """Base model never trained on calibration labels used by this meta-model."""
    cal=windows["calibration"]
    base=predictions["lightgbm"]
    y=(np.sign(base["calibration"])*cal.target_ticks.to_numpy()>config.cost_ticks).astype(int)
    if np.unique(y).size<2:
        return None,{key:np.zeros(len(value)) for key,value in base.items()}
    classifier=LGBMClassifier(n_estimators=100 if not config.quick else 30,num_leaves=7,
        max_depth=3,min_child_samples=80,learning_rate=0.03,reg_lambda=10.0,
        reg_alpha=1.0,subsample=.8,subsample_freq=1,colsample_bytree=.7,
        deterministic=True,force_col_wise=True,random_state=config.seed,
        n_jobs=config.threads,verbosity=-1)
    classifier.fit(meta_features(cal[features],base["calibration"],cal.scale_ticks.to_numpy(),config.cost_ticks),y)
    probabilities={}
    for name,values in base.items():
        frame=windows[name]
        probabilities[name]=classifier.predict_proba(meta_features(frame[features],values,frame.scale_ticks.to_numpy(),config.cost_ticks))[:,1]
    return classifier,probabilities


def forecast_metrics(frame,prediction):
    true=frame.target_ticks.to_numpy()
    active=np.abs(true)>0
    correlation=float(np.corrcoef(true,prediction)[0,1]) if np.std(prediction)>1e-12 else 0.0
    return {"mae_ticks":float(mean_absolute_error(true,prediction)),
            "rmse_ticks":float(np.sqrt(mean_squared_error(true,prediction))),
            "r2":float(r2_score(true,prediction)),"correlation":correlation,
            "direction_accuracy_nonzero":float((np.sign(true[active])==np.sign(prediction[active])).mean()) if active.any() else 0.0,
            "forecast_rows":len(frame)}


def feature_groups(features):
    groups={}
    for feature in features:
        if feature.startswith("ng_") or feature=="curve_available": group="gas_curve"
        elif feature.startswith("wti_"): group="wti"
        elif feature.startswith("lng_"): group="lng_equity"
        elif feature.startswith("eia_"): group="eia_calendar"
        elif feature.startswith(("minute_","dow_")): group="clock"
        elif "volume" in feature: group="volume"
        elif feature.startswith(("sigma_","vol_ratio_","atr_","range_vol")): group="volatility"
        else: group="price_position_momentum"
        groups.setdefault(group,[]).append(feature)
    return groups


def permutation_importance(models,validation,features,seed):
    """Joint feature-family block permutation on validation, not training gain.

    Perturbations can break market relationships; these are diagnostics, not
    coefficient significance tests or causal effects.
    """
    rng=np.random.default_rng(seed)
    x=validation[features]
    y=validation.target_ticks.to_numpy()
    scale=validation.scale_ticks.to_numpy()
    rows=[]
    for model_name in ("lightgbm","ebm"):
        model=models[model_name]
        original=np.sqrt(mean_squared_error(y,model.predict(x)*scale))
        for group,columns in feature_groups(features).items():
            deltas=[]
            blocks=[np.arange(i,min(i+6,len(x))) for i in range(0,len(x),6)]
            for _ in range(5):
                ix=np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
                perturbed=x.copy()
                perturbed.loc[:,columns]=x.iloc[ix][columns].to_numpy()
                error=np.sqrt(mean_squared_error(y,model.predict(perturbed)*scale))
                deltas.append(error-original)
            rows.append({"model":model_name,"feature_group":group,"rmse_increase_ticks":float(np.mean(deltas)),
                         "repeat_std":float(np.std(deltas)),"evaluation":"preceding validation only"})
    return rows


def run_fold(data,features,start,end,phase,horizon,config,output):
    fold=start.strftime("%Y-%m")
    windows=split_windows(data,start,end,config.train_days)
    log(f"{phase} {fold} h={horizon}: "+", ".join(f"{k}={len(v):,}" for k,v in windows.items()))
    sets={name:frame[features] for name,frame in windows.items() if name!="train"}
    began=time.monotonic()
    models,predictions,training=fit_predict_models(windows["train"][features],windows["train"].target_scaled.to_numpy(),
        windows["stop"][features],windows["stop"].target_scaled.to_numpy(),sets,
        seed=config.seed,threads=config.threads,quick=config.quick)
    log(f"{fold} h={horizon}: estimator suite fitted in {time.monotonic()-began:.1f}s")
    meta_model,meta_probability=fit_meta(windows,features,predictions,config)
    predictions["lightgbm_meta"]=predictions["lightgbm"]
    test=windows["test"]
    days=session_days(test)
    forecast_rows=[]; strategy_rows=[]; policy_rows=[]; selection_rows=[]; trade_frames=[]; daily_frames=[]; prediction_frames=[]
    policies={}
    for model_name in MODEL_NAMES:
        values=predictions[model_name]
        validation=windows["validation"]
        vp=values["validation"]*validation.scale_ticks.to_numpy()
        probability=meta_probability["validation"] if model_name=="lightgbm_meta" else None
        policy,trials=select_policy(validation,vp,config.cost_ticks,probability)
        policies[model_name]=policy
        prefix={"phase":phase,"fold":fold,"horizon":horizon,"model":model_name}
        policy_rows.append({**prefix,**policy})
        selection_rows.extend({**prefix,**trial} for trial in trials)
        predicted=values["test"]*test.scale_ticks.to_numpy()
        forecast_rows.append({**prefix,**forecast_metrics(test,predicted)})
        test_prob=meta_probability["test"] if model_name=="lightgbm_meta" else None
        # Freeze the selected signals at the selection cost, then stress costs.
        # Cost sensitivity never reselects thresholds or deletes expensive trades.
        trades=simulate(test,predicted,config.cost_ticks,policy["edge"],test_prob,policy["probability_floor"])
        for cost in sorted(set(config.stress_costs+(config.cost_ticks,))):
            stressed=trades.copy()
            stressed["cost_ticks"]=cost
            stressed["net_ticks"]=stressed.gross_ticks-cost
            strategy_rows.append({**prefix,"cost_ticks":cost,**metrics(stressed,days)})
        if not trades.empty:
            trade_frames.append(trades.assign(**prefix))
        daily=daily_pnl(trades,days)
        daily_frames.append(pd.DataFrame({"date":days,"net_ticks":daily.to_numpy(),**prefix}))
        prediction_frames.append(test[["decision_time","entry_time","exit_time","target_ticks","scale_ticks"]].assign(
            **prefix,predicted_ticks=predicted,probability_net_win=(test_prob if test_prob is not None else np.nan)))
    forecast_rows.append({"phase":phase,"fold":fold,"horizon":horizon,"model":"zero_change",**forecast_metrics(test,np.zeros(len(test)))})

    # Basic rules share the same timing, costs and validation gate as ML.
    for model_name,column,direction in (("always_long",None,1),("momentum_rule","momentum_15m_vol",1),("reversion_rule","momentum_15m_vol",-1)):
        validation=windows["validation"]
        vp=np.full(len(validation),1000.0) if column is None else direction*np.sign(validation[column].fillna(0).to_numpy())*1000
        policy,trials=select_policy(validation,vp,config.cost_ticks)
        predicted=np.full(len(test),1000.0) if column is None else direction*np.sign(test[column].fillna(0).to_numpy())*1000
        trades=simulate(test,predicted,config.cost_ticks,policy["edge"])
        prefix={"phase":phase,"fold":fold,"horizon":horizon,"model":model_name}
        policies[model_name]=policy
        policy_rows.append({**prefix,**policy})
        selection_rows.extend({**prefix,**trial} for trial in trials)
        for cost in sorted(set(config.stress_costs+(config.cost_ticks,))):
            stressed=trades.copy(); stressed["cost_ticks"]=cost; stressed["net_ticks"]=stressed.gross_ticks-cost
            strategy_rows.append({**prefix,"cost_ticks":cost,**metrics(stressed,days)})
        if not trades.empty: trade_frames.append(trades.assign(**prefix))
        daily=daily_pnl(trades,days)
        daily_frames.append(pd.DataFrame({"date":days,"net_ticks":daily.to_numpy(),**prefix}))

    # Calibrate q10/q90 interval expansion on a chronological calibration set.
    cal=windows["calibration"]
    qlo=np.minimum(predictions["lightgbm_q10"]["calibration"],predictions["lightgbm_q90"]["calibration"])
    qhi=np.maximum(predictions["lightgbm_q10"]["calibration"],predictions["lightgbm_q90"]["calibration"])
    residual=np.maximum(qlo-cal.target_scaled.to_numpy(),cal.target_scaled.to_numpy()-qhi)
    quantile=min(1.0,np.ceil((len(residual)+1)*.8)/len(residual))
    expansion=max(0.0,float(np.quantile(residual,quantile,method="higher")))
    lo=(np.minimum(predictions["lightgbm_q10"]["test"],predictions["lightgbm_q90"]["test"])-expansion)*test.scale_ticks.to_numpy()
    hi=(np.maximum(predictions["lightgbm_q10"]["test"],predictions["lightgbm_q90"]["test"])+expansion)*test.scale_ticks.to_numpy()
    interval={"phase":phase,"fold":fold,"horizon":horizon,"nominal_coverage":.8,
              "observed_coverage":float(((test.target_ticks>=lo)&(test.target_ticks<=hi)).mean()),
              "mean_width_ticks":float(np.mean(hi-lo)),"calibration_expansion_scaled":expansion}
    base_dir=output/f"{phase}_{fold}_{horizon}m"
    base_dir.mkdir()
    for name,rows in (("forecast_metrics",forecast_rows),("strategy_metrics",strategy_rows),("policies",policy_rows),
                       ("validation_trials",selection_rows),("training_metadata",training)):
        pd.DataFrame(rows).to_csv(base_dir/f"{name}.csv",index=False)
    preds=pd.concat(prediction_frames,ignore_index=True)
    preds.to_parquet(base_dir/"predictions.parquet",index=False)
    trades=pd.concat(trade_frames,ignore_index=True) if trade_frames else pd.DataFrame(columns=["phase","fold","horizon","model","entry_time","exit_time","net_ticks","gross_ticks","cost_ticks"])
    trades.to_csv(base_dir/"trades.csv",index=False)
    daily=pd.concat(daily_frames,ignore_index=True)
    daily.to_csv(base_dir/"daily_pnl.csv",index=False)
    json_write(base_dir/"interval_diagnostic.json",interval)
    if phase=="holdout":
        joblib.dump({"models":models,"meta_model":meta_model,"policies":policies,"features":features,
                     "config":asdict(config),"horizon":horizon,"trained_labels_through":str(windows["train"].exit_time.max()),
                     "policy_selected_through":str(windows["validation"].exit_time.max()),
                     "interval_expansion_scaled":expansion,"deployment_enabled":False,
                     "purpose":"frozen retrospective research model"},base_dir/"models.joblib",compress=3)
        importance=permutation_importance(models,windows["validation"],features,config.seed)
        pd.DataFrame(importance).to_csv(base_dir/"validation_group_importance.csv",index=False)
        ebm=models["ebm"].estimator
        pd.DataFrame({"term":ebm.term_names_,"importance":ebm.term_importances()}).sort_values("importance",ascending=False).to_csv(base_dir/"ebm_terms.csv",index=False)
    log(f"{phase} {fold} h={horizon}: "+", ".join(f"{p['model']} {'on' if p['enabled'] else 'off'}" for p in policy_rows))
    return {"forecasts":forecast_rows,"strategies":strategy_rows,"policies":policy_rows,"daily":daily,"trades":trades,"interval":interval}


def summarize(results,config):
    daily=pd.concat([r["daily"] for r in results],ignore_index=True)
    trades=pd.concat([r["trades"] for r in results],ignore_index=True)
    rows=[]
    for (phase,horizon,model),group in daily.groupby(["phase","horizon","model"]):
        candidate=trades.loc[trades.phase.eq(phase)&trades.horizon.eq(horizon)&trades.model.eq(model)].copy()
        if not candidate.empty:
            candidate["entry_time"]=pd.to_datetime(candidate.entry_time,utc=True)
            candidate=candidate.sort_values("entry_time")
        for cost in sorted(set(config.stress_costs+(config.cost_ticks,))):
            candidate["net_ticks"]=candidate.gross_ticks-cost
            stats=metrics(candidate,pd.DatetimeIndex(group.date))
            blocks=block_bootstrap(daily_pnl(candidate,pd.DatetimeIndex(group.date)).to_numpy(),seed=config.seed)
            rows.append({"phase":phase,"horizon":horizon,"model":model,"cost_ticks":cost,**stats,**blocks})
    return pd.DataFrame(rows),daily,trades


def select_research_candidate(summary,config):
    dev=summary.loc[summary.phase.eq("development")&summary.cost_ticks.eq(config.cost_ticks)].copy()
    # A profitable confidence lower bound in development is required to select
    # any policy for the subsequent holdout. Cash remains a legitimate result.
    eligible=dev.loc[dev.trades.ge(40)&dev.mean_daily_ci_low.gt(0)]
    if eligible.empty:
        return {"model":"cash","horizon":None,"reason":"No development candidate passed a positive block-bootstrap lower bound with at least 40 trades"}
    winner=eligible.sort_values(["mean_daily_ci_low","net_ticks"],ascending=False).iloc[0]
    return {"model":winner.model,"horizon":int(winner.horizon),"reason":"Largest development lower confidence bound among candidates with at least 40 trades"}


def write_report(output,config,summary,forecast,selection,checks,audit,intervals):
    cost=summary.loc[summary.cost_ticks.eq(config.cost_ticks)]
    table_columns=["horizon","model","trades","net_ticks","mean_net_ticks","profit_factor","max_drawdown_ticks","daily_sharpe","mean_daily_ci_low","mean_daily_ci_high"]
    final=cost.loc[cost.phase.eq("holdout")].sort_values(["horizon","model"])
    dev=cost.loc[cost.phase.eq("development")].sort_values("net_ticks",ascending=False)
    if selection["model"]=="cash":
        decision="The predeclared selection rule chose CASH before holdout predictions were evaluated. The selected strategy therefore made no trades and earned zero; other holdout results below are benchmarks, not retrospectively selected winners."
    else:
        chosen=final.loc[final.model.eq(selection["model"])&final.horizon.eq(selection["horizon"])].iloc[0]
        decision=(f"Selected before holdout: {selection['model']} at {selection['horizon']} minutes. "
                  f"Holdout: {int(chosen.trades)} trades, {chosen.net_ticks:.1f} net ticks at {config.cost_ticks:g} ticks cost, "
                  f"daily mean 95% block interval [{chosen.mean_daily_ci_low:.2f}, {chosen.mean_daily_ci_high:.2f}] ticks.")
    lines=["# Advanced natural gas modelling experiment","",decision,"",
        "## Scope and execution","",
        f"- {audit['raw_rows']:,} input bars, {audit['features']} features; horizons {config.horizons} minutes.",
        "- Features available after each minute closes; decision at bar start +1 minute, entry at +2 minutes, exit at entry+horizon using open prices.",
        "- Integer tick labels; entire path must be contiguous, same contract and good data quality. Auxiliary returns reset on their own rolls.",
        "- 08:00–16:00 New York, five-minute decision grid, one contract/position at a time; positions excluded around scheduled EIA releases.",
        "- EIA calendar retained; historical storage actuals excluded because their publication vintages could not be fully verified.",
        f"- Fixed selection cost {config.cost_ticks:g} ticks round trip. Cost sensitivity preserves original trades; $10/tick is a full-size NG assumption, not MNG.",
        "- Training (most recent 180 days), separate chronological early-stop, meta/calibration, policy-validation and subsequent test partitions; labels purged at every boundary.",
        "- EBM uses fixed-round additive boosting plus interactions, no internal random validation. Random forest OOB scores are not used.",
        "- Volatility-normalized targets; models only clip training targets at +/-8 scale units. Forecast errors and trading P&L use uncapped actual moves.",
        "- LightGBM meta-filter learns net-win labels only from base forecasts on a later calibration partition. Equal-weight ensemble has no test-fitted weights.",
        "- Four cost-edge thresholds per model; meta-filter also tests three probability thresholds. Validation requires at least 20 trades, five active days and positive results in both halves.",
        "- Development: May–July 2026 by default. Final Aug–Sep period withheld from this run's model selection; models/policies frozen before it starts.",
        "", "## Development results", "", dev[table_columns].round(4).to_markdown(index=False),
        "", "## Final historical evaluation", "", final[table_columns].round(4).to_markdown(index=False),
        "", "## Forecasting versus zero change", "",forecast.round(4).to_markdown(index=False),
        "", "## Uncertainty and multiple comparisons", "",
        f"Centered block-bootstrap max-mean family tests: `{checks}`.",
        "Five-session circular blocks with 2,000 resamples include zero-trade days. These are dependence-aware diagnostics, not a guarantee of statistical significance. The family test covers the registered candidate strategies in this run, not every past/adaptive research decision.",
        "",pd.DataFrame(intervals).round(4).to_markdown(index=False),
        "", "Intervals are LightGBM q10/q90 forecasts widened using prior calibration residuals; 80% coverage is a target, not a time-series guarantee.",
        "", "## Limitations and decision", "",
        "This is exploratory historical research: these dates were already inspected in the original model. New model-selection boundaries do not turn old data into a fresh independent test. No live deployment is enabled, even if a benchmark is profitable.",
        "OHLCV has no bid/ask quotes, market depth or actual fills. The delayed-open simulation and fixed cost scenarios remain approximations. Models do not add information missing from the inputs.",
        "Results condition on complete future price paths: outages and missing bars are excluded after the fact. Zero-trade days are included when that session contains eligible observations; days with no eligible rows are outside this sample. A live missing-data/exit policy is not simulated.",
        "Feature gain and EBM term magnitude are not p-values. `validation_group_importance.csv` measures held-out loss changes under joint block permutations; negative/near-zero values do not establish useful prediction.",
        "Next evidential step is frozen paper trading/new history with measured execution costs; do not keep optimizing this final period until its P&L becomes positive.",
        "", "Research rationale and primary references: [ADVANCED_MODEL_RESEARCH.md](../ADVANCED_MODEL_RESEARCH.md).", ""]
    (output/"advanced_report.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--output",type=Path,default=PROJECT/"model_output_advanced")
    parser.add_argument("--horizons",type=int,nargs="+",default=[5,15,30])
    parser.add_argument("--cost-ticks",type=float,default=2.0)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--quick",action="store_true",help="Reduced estimator budgets for a complete smoke run")
    parser.add_argument("--development-start",default="2026-05-01")
    parser.add_argument("--holdout-start",default="2026-08-01")
    args=parser.parse_args()
    config=Config(horizons=tuple(args.horizons),cost_ticks=args.cost_ticks,threads=args.threads,quick=args.quick)
    if config.cost_ticks<0 or config.threads<1 or any(h<1 for h in config.horizons):
        parser.error("Costs must be nonnegative; horizons and threads positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve the experiment record")
    args.output.mkdir(parents=True)
    development_start=pd.Timestamp(args.development_start,tz="UTC")
    holdout_start=pd.Timestamp(args.holdout_start,tz="UTC")
    if development_start>=holdout_start:
        parser.error("Development must precede holdout")
    manifest={"config":asdict(config),"created_utc":datetime.now(timezone.utc).isoformat(),
              "data":str(args.data.resolve()),"data_sha256":file_hash(args.data),
              "development_start":str(development_start),"holdout_start":str(holdout_start),
              "historical_dates_previously_inspected":True,"deployment_enabled":False,
              "packages":{p:importlib.metadata.version(p) for p in ("lightgbm","scikit-learn","interpret-core","numpy","pandas")},
              "source_sha256":{p:file_hash(PROJECT/p) for p in ("train_advanced.py","advanced_data.py","advanced_backtest.py","advanced_estimators.py")},
              "candidate_models":list(MODEL_NAMES)+["always_long","momentum_rule","reversion_rule"],
              "policy_grid":{"edge_ticks":[0,1,2,4],"meta_probability":[.5,.6,.7]}}
    json_write(args.output/"manifest.json",manifest)
    frame,features,audit=prepare_features(pd.read_parquet(args.data))
    datasets={h:executable_dataset(frame,h,delay_bars=config.delay_bars,stride=config.sample_stride) for h in config.horizons}
    audit["eligible_rows_by_horizon"]={h:len(d) for h,d in datasets.items()}
    json_write(args.output/"data_audit.json",audit)
    json_write(args.output/"feature_list.json",features)
    log(f"Prepared {len(features)} causal features; eligible rows {audit['eligible_rows_by_horizon']}")
    results=[]
    starts=pd.date_range(development_start,holdout_start,freq="MS",inclusive="left")
    for start in starts:
        for horizon,data in datasets.items():
            results.append(run_fold(data,features,start,start+pd.DateOffset(months=1),"development",horizon,config,args.output))
    development_summary,_,_=summarize(results,config)
    selection=select_research_candidate(development_summary,config)
    # This file is persisted BEFORE computing or inspecting ANY final forecasts.
    json_write(args.output/"frozen_selection.json",{**selection,"frozen_utc":datetime.now(timezone.utc).isoformat()})
    log(f"Frozen development selection: {selection}")
    end=frame.ts_event.max()+pd.Timedelta(minutes=1)
    for horizon,data in datasets.items():
        results.append(run_fold(data,features,holdout_start,end,"holdout",horizon,config,args.output))
    summary,daily,trades=summarize(results,config)
    summary.to_csv(args.output/"strategy_summary.csv",index=False)
    daily.to_csv(args.output/"daily_pnl.csv",index=False)
    trades.to_csv(args.output/"all_trades.csv",index=False)
    forecasts=pd.DataFrame([row for r in results for row in r["forecasts"]])
    forecasts.to_csv(args.output/"forecast_by_fold.csv",index=False)
    forecast_summary=forecasts.groupby(["phase","horizon","model"],as_index=False)[["mae_ticks","rmse_ticks","r2","correlation","direction_accuracy_nonzero"]].mean()
    forecast_summary.to_csv(args.output/"forecast_summary.csv",index=False)
    checks={}
    for phase,group in daily.groupby("phase"):
        matrix=group.pivot_table(index="date",columns=["horizon","model"],values="net_ticks",aggfunc="sum").fillna(0)
        checks[phase]=reality_check(matrix.to_numpy(),seed=config.seed)
    json_write(args.output/"family_tests.json",checks)
    write_report(args.output,config,summary,forecast_summary,selection,checks,audit,[r["interval"] for r in results])
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,(phase,group) in zip(axes,daily.groupby("phase",sort=True)):
        matrix=group.pivot_table(index="date",columns=["horizon","model"],values="net_ticks",aggfunc="sum").fillna(0)
        for (horizon,model) in matrix:
            if matrix[(horizon,model)].abs().sum()>0:
                ax.plot(matrix.index,matrix[(horizon,model)].cumsum(),label=f"{model} {horizon}m")
        ax.axhline(0,color="black",lw=.8); ax.set_title(phase); ax.set_ylabel("Cumulative net ticks at fixed selection cost")
        if ax.lines and len(ax.lines)>1: ax.legend(fontsize=6)
        ax.tick_params(axis="x",rotation=30)
    fig.tight_layout(); fig.savefig(args.output/"equity_curves.png",dpi=150); plt.close(fig)
    log(f"Complete: {args.output/'advanced_report.md'}")


if __name__=="__main__":
    main()
