"""Small daily models and chronological forecasts for intraday conditioning.

Each daily observation is used once. A monthly model sees only labels completed
before that month starts. The final evaluation uses one frozen daily model,
although its historical indicator inputs continue updating each morning.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DAILY_SCORE_FEATURES = tuple(
    f"daily_{name}_{h}d" for h in (1, 3)
    for name in ("expected_log_return", "expected_scaled", "p_down", "model_available")
)


@dataclass
class DailyModel:
    features: tuple[str, ...]
    horizon: int
    regressor: object | None
    classifier: object | None
    fallback_scale: float
    base_p_down: float
    training_rows: int
    cutoff: pd.Timestamp
    last_label_available: pd.Timestamp | None

    def predict(self, daily: pd.DataFrame) -> pd.DataFrame:
        scale = daily.ctx_rms_day_return20.fillna(self.fallback_scale).clip(lower=.002).to_numpy()*np.sqrt(self.horizon)
        available = self.regressor is not None
        if available:
            expected_scaled = np.clip(self.regressor.predict(daily[list(self.features)]), -3., 3.)
            p_down = self.classifier.predict_proba(daily[list(self.features)])[:, 1] if self.classifier is not None else np.full(len(daily),self.base_p_down)
        else:
            expected_scaled = np.zeros(len(daily))
            p_down = np.full(len(daily),self.base_p_down)
        return pd.DataFrame({
            "session_date":daily.session_date.to_numpy(),
            f"daily_expected_log_return_{self.horizon}d":expected_scaled*scale,
            f"daily_expected_scaled_{self.horizon}d":expected_scaled,
            f"daily_p_down_{self.horizon}d":p_down,
            f"daily_model_available_{self.horizon}d":float(available),
            f"daily_base_p_down_{self.horizon}d":self.base_p_down,
            f"daily_train_rows_{self.horizon}d":self.training_rows,
            f"daily_fit_cutoff_{self.horizon}d":self.cutoff,
        })


def fit_daily_model(daily: pd.DataFrame, cutoff: pd.Timestamp, horizon: int,
                    *, min_train: int=40, seed: int=42) -> DailyModel:
    cutoff=pd.Timestamp(cutoff)
    cutoff=cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    features=tuple(c for c in daily if c.startswith("ctx_"))
    label=f"future_{horizon}d_log_return"
    availability=f"future_{horizon}d_available_at"
    # Actual outcome completion is the purge, not a fixed number of row shifts.
    train=daily.loc[daily[availability].lt(cutoff)&daily[label].notna()&daily.snapshot_at.lt(cutoff)].copy()
    fallback=max(.002,float(train.future_1d_log_return.std())) if len(train)>1 else .02
    if not np.isfinite(fallback): fallback=.02
    rate=float(train[label].lt(0).mean()) if len(train) else .5
    result=DailyModel(features,horizon,None,None,fallback,rate,len(train),cutoff,
                      train[availability].max() if len(train) else None)
    if len(train)<min_train:
        return result
    scale=train.ctx_rms_day_return20.fillna(fallback).clip(lower=.002).to_numpy()*np.sqrt(horizon)
    target=np.clip(train[label].to_numpy()/scale,-4,4)
    result.regressor=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                                  StandardScaler(),Ridge(alpha=100.0))
    result.regressor.fit(train[list(features)],target)
    direction=train[label].lt(0).astype(int)
    if direction.nunique()>1:
        result.classifier=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
            StandardScaler(),LogisticRegression(C=.05,max_iter=2000,random_state=seed))
        result.classifier.fit(train[list(features)],direction)
    return result


def prequential_daily_forecasts(daily: pd.DataFrame, *, holdout_start: str | pd.Timestamp="2026-08-01",
                                 min_train: int=40, seed: int=42) -> tuple[pd.DataFrame,dict,list[dict]]:
    """Out-of-training monthly predictions; freeze all final-period months.

    No fit ever receives a contemporaneous or later realised daily target.
    Generated scores for early sessions use a zero-mean fallback until 40 labels
    have matured. The fallback's availability flag is an explicit feature.
    """
    source=daily.sort_values("snapshot_at").reset_index(drop=True)
    freeze=pd.Timestamp(holdout_start)
    freeze=freeze.tz_localize("UTC") if freeze.tzinfo is None else freeze.tz_convert("UTC")
    months=source.session_date.dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    cutoffs=months.where(months.lt(freeze),freeze)
    parts=[]; bundles={}; metadata=[]
    for cutoff,group in source.groupby(cutoffs,sort=True):
        pair=[]
        for horizon in (1,3):
            model=fit_daily_model(source,cutoff,horizon,min_train=min_train,seed=seed)
            scored=model.predict(group)
            pair.append(scored)
            bundles[(str(cutoff),horizon)]=model
            metadata.append({"cutoff":str(cutoff),"horizon_days":horizon,"training_rows":model.training_rows,
                             "label_available_through":str(model.last_label_available),
                             "features":len(model.features),"model_available":model.regressor is not None,
                             "prediction_sessions":len(group),"frozen_final_period":bool(cutoff==freeze)})
        parts.append(pair[0].merge(pair[1],on="session_date",validate="one_to_one"))
    scores=pd.concat(parts,ignore_index=True).sort_values("session_date").reset_index(drop=True)
    return scores,bundles,metadata


def join_daily_forecasts(frame: pd.DataFrame,scores: pd.DataFrame) -> pd.DataFrame:
    result=frame.copy()
    dates=result.decision_time.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    lookup=scores.set_index("session_date")
    local=result.decision_time.dt.tz_convert("America/New_York")
    known=local.dt.hour.ge(8)
    for name in DAILY_SCORE_FEATURES:
        result[name]=dates.map(lookup[name]).where(known).astype("float32")
    return result


@dataclass
class ReprojectedRegressor:
    base: object
    correction: object | None
    signal_features: tuple[str,...] = ("daily_expected_scaled_1d","daily_expected_scaled_3d")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base=self.base.predict(frame)
        adjustment=(self.correction.predict(frame[list(self.signal_features)])
                    if self.correction is not None else np.zeros(len(frame)))
        return base+adjustment


def fit_reprojection(base,calibration: pd.DataFrame) -> ReprojectedRegressor:
    """Learn daily outlook -> short-horizon residual, with ONE weight per day."""
    signal_features=("daily_expected_scaled_1d","daily_expected_scaled_3d")
    data=calibration[list(signal_features)].copy()
    data["residual"]=calibration.target_scaled.to_numpy()-base.predict(calibration)
    data["session"]=calibration.decision_time.dt.tz_convert("America/New_York").dt.date
    by_day=data.groupby("session").mean()
    model=ReprojectedRegressor(base,None)
    if len(by_day)>=5:
        correction=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                                 StandardScaler(),Ridge(alpha=20.0))
        correction.fit(by_day[list(signal_features)],by_day.residual)
        model.correction=correction
    return model


@dataclass
class DailyDriftRegressor:
    horizon_minutes: int

    def predict(self,frame: pd.DataFrame) -> np.ndarray:
        # Transparent benchmark assumption: the expected day move accrues
        # uniformly over the 480-minute research session. Not an arbitrage law.
        move=frame.close.to_numpy()*np.expm1(frame.daily_expected_log_return_1d.fillna(0).to_numpy()*self.horizon_minutes/480)/.001
        scale=(frame.sigma_1m_ticks*np.sqrt(self.horizon_minutes)).clip(lower=2).to_numpy()
        return move/scale
