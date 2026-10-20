"""Cost-aware policy selection, non-overlapping fills and block uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(frame: pd.DataFrame, prediction: np.ndarray, cost: float,
             edge: float, probability: np.ndarray | None=None,
             probability_floor: float=0.0) -> pd.DataFrame:
    prediction = np.asarray(prediction,dtype=float)
    if len(prediction) != len(frame):
        raise ValueError("Predictions and execution rows must align")
    signal = np.where(np.abs(prediction) > cost+edge,np.sign(prediction),0).astype(int)
    if probability is not None:
        signal[np.asarray(probability) < probability_floor] = 0
    entry_ns = frame.entry_time.astype("int64").to_numpy()
    exit_ns = frame.exit_time.astype("int64").to_numpy()
    selected = []
    next_entry = np.iinfo(np.int64).min
    for i in np.flatnonzero(signal):
        if entry_ns[i] >= next_entry:
            selected.append(i)
            next_entry = exit_ns[i]
    columns = ["decision_time","entry_time","exit_time","entry_price","exit_price","target_ticks"]
    trades = frame.iloc[selected][columns].copy().reset_index(drop=True)
    trades["direction"] = signal[selected]
    trades["predicted_ticks"] = prediction[selected]
    trades["gross_ticks"] = trades.direction*trades.target_ticks
    trades["cost_ticks"] = cost
    trades["net_ticks"] = trades.gross_ticks-cost
    return trades


def session_days(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(frame.entry_time.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize().unique()).sort_values()


def daily_pnl(trades: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    if trades.empty:
        return pd.Series(0.0,index=days)
    dates = trades.entry_time.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return trades.groupby(dates).net_ticks.sum().reindex(days,fill_value=0.0)


def metrics(trades: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    daily = daily_pnl(trades,days)
    net = trades.net_ticks.to_numpy()
    equity = np.r_[0.0,np.cumsum(net)]
    loss = -net[net<0].sum()
    win = net[net>0].sum()
    return {"trades":len(trades),"net_ticks":float(net.sum()),"net_usd_per_ng_contract":float(net.sum()*10),
            "mean_net_ticks":float(net.mean()) if len(net) else 0.0,
            "win_rate":float((net>0).mean()) if len(net) else 0.0,
            "profit_factor":float(win/loss) if loss>0 else (float("nan") if not len(net) else float("inf")),
            "max_drawdown_ticks":float((equity-np.maximum.accumulate(equity)).min()),
            "daily_sharpe":float(np.sqrt(252)*daily.mean()/daily.std()) if len(daily)>1 and daily.std()>0 else 0.0,
            "active_days":int(daily.ne(0).sum()),"evaluated_days":len(daily)}


def select_policy(frame: pd.DataFrame,prediction: np.ndarray,cost: float,
                  probability: np.ndarray | None=None) -> tuple[dict,list[dict]]:
    """Predeclared small grid; select only on the preceding validation window."""
    days = session_days(frame)
    candidates = []
    for edge in (0.0,1.0,2.0,4.0):
        for floor in ((0.50,0.60,0.70) if probability is not None else (0.0,)):
            trades = simulate(frame,prediction,cost,edge,probability,floor)
            stats = metrics(trades,days)
            daily = daily_pnl(trades,days)
            half = len(daily)//2
            se = daily.std()/np.sqrt(len(daily)) if len(daily)>1 else float("inf")
            score = float(daily.mean()-se)
            eligible = (stats["trades"]>=20 and stats["active_days"]>=5 and score>0
                        and daily.iloc[:half].sum()>0 and daily.iloc[half:].sum()>0)
            candidates.append({"edge":edge,"probability_floor":floor,"enabled":bool(eligible),
                               "validation_score":score,**stats})
    eligible = [c for c in candidates if c["enabled"]]
    if eligible:
        return max(eligible,key=lambda c:c["validation_score"]),candidates
    return {"edge":float("inf"),"probability_floor":1.0,"enabled":False,
            "validation_score":0.0,"reason":"No validation policy passed costs, minimum activity and both-half stability"},candidates


def block_bootstrap(daily: np.ndarray,*,samples: int=2000,block: int=5,seed: int=42) -> dict:
    """Circular five-session blocks; descriptive CI, not proof after model search."""
    x = np.asarray(daily,dtype=float)
    if not len(x):
        return {"mean_daily_ci_low":0.0,"mean_daily_ci_high":0.0,"bootstrap_positive_fraction":0.0}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,len(x),size=(samples,int(np.ceil(len(x)/block))))
    indexes = (starts[:,:,None]+np.arange(block)) % len(x)
    means = x[indexes.reshape(samples,-1)[:,:len(x)]].mean(axis=1)
    low,high = np.quantile(means,[0.025,0.975])
    return {"mean_daily_ci_low":float(low),"mean_daily_ci_high":float(high),
            "bootstrap_positive_fraction":float((means>0).mean())}


def reality_check(daily_matrix: np.ndarray,*,samples: int=2000,seed: int=42,block: int=5) -> dict:
    """Centered block-bootstrap maximum mean across registered candidate rules.

    Conservative White-style family test; not a full SPA/DSR implementation.
    Uses zero-position cash as reference and preserves common market days.
    """
    x = np.asarray(daily_matrix,dtype=float)
    if x.ndim!=2 or not x.shape[0] or not x.shape[1]:
        return {"family_p_value":1.0,"candidate_count":0}
    observed = max(0.0,float(x.mean(axis=0).max()))
    if observed<=0:
        return {"family_p_value":1.0,"candidate_count":x.shape[1]}
    centered = x-x.mean(axis=0)
    rng = np.random.default_rng(seed)
    count=0
    for _ in range(samples):
        starts = rng.integers(0,len(x),size=int(np.ceil(len(x)/block)))
        ix = ((starts[:,None]+np.arange(block))%len(x)).ravel()[:len(x)]
        count += float(centered[ix].mean(axis=0).max()) >= observed
    return {"family_p_value":float((count+1)/(samples+1)),"candidate_count":x.shape[1]}
