"""Regularized, reproducible estimators for chronological trading experiments.

All returned predictions are in the caller's volatility-normalized target units.
The caller owns time splits, target scaling, trading rules, and untouched tests.
Only LightGBM uses ``x_stop``: EBM has fixed rounds and no random validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STRATEGY_MODELS = (
    "lightgbm", "ebm", "random_forest", "extra_trees", "ridge", "ensemble"
)
QUANTILE_MODELS = ("lightgbm_q10", "lightgbm_q90")
ENSEMBLE_MEMBERS = ("lightgbm", "ebm", "random_forest", "extra_trees")


def _numeric_values(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """Align named columns and treat nonfinite feature values as missing."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Features must be provided as a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("Feature column names must be unique.")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    values = frame.loc[:, list(features)].to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass
class FittedRegressor:
    """Picklable estimator with training-only imputation and column alignment.

    ``estimator`` exposes the underlying model, including EBM term explanations.
    """

    feature_names: tuple[str, ...]
    imputer: SimpleImputer
    estimator: Any

    @property
    def model(self) -> Any:
        return self.estimator

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_values(frame, self.feature_names)
        if not len(values):
            return np.empty(0, dtype=np.float64)
        return self.predict_imputed(self.imputer.transform(values))

    def predict_imputed(self, values: np.ndarray) -> np.ndarray:
        if not len(values):
            return np.empty(0, dtype=np.float64)
        # LightGBM records feature names even when fitted from an ndarray.
        # A named frame also lets its sklearn API verify the column schema.
        model_input = (
            pd.DataFrame(values, columns=self.feature_names)
            if isinstance(self.estimator, LGBMRegressor)
            else values
        )
        return np.asarray(self.estimator.predict(model_input), dtype=np.float64)


@dataclass
class MeanEnsemble:
    """An equal-weight ensemble whose weights never see validation/test P&L."""

    members: tuple[FittedRegressor, ...]
    member_names: tuple[str, ...] = ENSEMBLE_MEMBERS

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.members:
            raise ValueError("The ensemble must contain at least one estimator.")
        return np.mean([member.predict(frame) for member in self.members], axis=0)


def _target(values: np.ndarray, row_count: int, name: str) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64)
    if target.ndim != 1 or len(target) != row_count or not len(target):
        raise ValueError(f"{name} must be a nonempty 1-D array matching its features.")
    if not np.isfinite(target).all():
        raise ValueError(f"{name} contains missing or infinite targets.")
    return target


def fit_predict_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_stop: pd.DataFrame,
    y_stop: np.ndarray,
    prediction_sets: dict[str, pd.DataFrame],
    *,
    seed: int = 42,
    threads: int = 4,
    quick: bool = False,
) -> tuple[
    dict[str, FittedRegressor | MeanEnsemble],
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    """Fit six strategy candidates and two diagnostic quantile regressors.

    ``x_stop`` must precede strategy calibration and test periods, with any
    overlapping target intervals purged by the caller. Its targets remain
    unclipped for early stopping; only training targets are clipped to [-8, 8].
    No estimator, imputer, scaling transform, or EBM bin sees prediction sets
    during fitting. An unavailable EBM installation is an error, never a skip.
    """
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
    except ImportError as exc:
        raise ImportError(
            "EBM is required for this experiment. Install interpret-core in the "
            "project environment before running advanced training."
        ) from exc

    if not isinstance(x_train, pd.DataFrame) or x_train.shape[1] == 0:
        raise ValueError("x_train must be a DataFrame with at least one feature.")
    if any(not isinstance(name, str) for name in x_train.columns):
        raise ValueError("Feature names must be strings.")
    if threads < 1:
        raise ValueError("threads must be a positive integer.")
    features = tuple(x_train.columns)
    raw_y_train = _target(y_train, len(x_train), "y_train")
    stop_target = _target(y_stop, len(x_stop), "y_stop")
    train_target = np.clip(raw_y_train, -8.0, 8.0)

    preprocessing_start = perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_values = imputer.fit_transform(_numeric_values(x_train, features))
    stop_values = imputer.transform(_numeric_values(x_stop, features))
    prediction_values: dict[str, np.ndarray] = {}
    for name, frame in prediction_sets.items():
        values = _numeric_values(frame, features)
        prediction_values[name] = imputer.transform(values) if len(values) else values
    preprocessing_seconds = perf_counter() - preprocessing_start

    lgb_parameters = dict(
        n_estimators=100 if quick else 600,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=200,
        min_split_gain=0.01,
        reg_alpha=1.0,
        reg_lambda=10.0,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        max_bin=63,
        deterministic=True,
        force_col_wise=True,
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
        importance_type="gain",
    )
    forest_parameters = dict(
        n_estimators=40 if quick else 160,
        max_depth=10,
        min_samples_leaf=100,
        max_features=0.7,
        random_state=seed,
        n_jobs=threads,
    )
    estimators = {
        "lightgbm": LGBMRegressor(objective="regression", **lgb_parameters),
        "ebm": ExplainableBoostingRegressor(
            feature_names=list(features),
            feature_types=["continuous"] * len(features),
            max_bins=32 if quick else 64,
            max_interaction_bins=16 if quick else 32,
            interactions=min(8, len(features) * (len(features) - 1) // 2),
            validation_size=0,
            outer_bags=1,
            inner_bags=0,
            max_rounds=40 if quick else 400,
            early_stopping_rounds=0,
            learning_rate=0.025,
            greedy_ratio=0.0,
            smoothing_rounds=10 if quick else 100,
            interaction_smoothing_rounds=10 if quick else 50,
            min_samples_leaf=200,
            reg_alpha=1.0,
            reg_lambda=10.0,
            max_leaves=2,
            max_delta_step=0.5,
            n_jobs=threads,
            random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            bootstrap=True, max_samples=0.75, **forest_parameters
        ),
        "extra_trees": ExtraTreesRegressor(bootstrap=False, **forest_parameters),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "lightgbm_q10": LGBMRegressor(
            objective="quantile", alpha=0.1, **lgb_parameters
        ),
        "lightgbm_q90": LGBMRegressor(
            objective="quantile", alpha=0.9, **lgb_parameters
        ),
    }
    artifacts: dict[str, FittedRegressor | MeanEnsemble] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    metadata: list[dict[str, Any]] = []
    for name, estimator in estimators.items():
        started = perf_counter()
        print(f"  Fitting {name} ({len(train_target):,} training rows)...", flush=True)
        if isinstance(estimator, LGBMRegressor):
            estimator.fit(
                pd.DataFrame(train_values, columns=features),
                train_target,
                eval_set=[(pd.DataFrame(stop_values, columns=features), stop_target)],
                eval_metric="quantile" if name in QUANTILE_MODELS else "l2",
                callbacks=[
                    early_stopping(20 if quick else 40, first_metric_only=True, verbose=False),
                    log_evaluation(0),
                ],
            )
            iterations: Any = int(estimator.best_iteration_ or estimator.n_estimators_)
        else:
            estimator.fit(train_values, train_target)
            if name == "ebm":
                iterations = np.asarray(estimator.best_iteration_).tolist()
            elif name in ("random_forest", "extra_trees"):
                iterations = int(estimator.n_estimators)
            else:
                iterations = None
        fit_seconds = perf_counter() - started
        artifact = FittedRegressor(features, imputer, estimator)
        artifacts[name] = artifact
        predictions[name] = {
            set_name: artifact.predict_imputed(values)
            for set_name, values in prediction_values.items()
        }
        elapsed_seconds = perf_counter() - started
        metadata.append({
            "model": name,
            "strategy_candidate": name not in QUANTILE_MODELS,
            "train_rows": len(train_target),
            "stop_rows": len(stop_target),
            "features": len(features),
            "fit_seconds": fit_seconds,
            "elapsed_seconds": elapsed_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "iterations": iterations,
            "training_target_clip": [-8.0, 8.0],
            "training_target_clipped_count": int(np.sum(raw_y_train != train_target)),
            "stop_target_clipped": False,
            "uses_stop_set": isinstance(estimator, LGBMRegressor),
            "quick": quick,
            "seed": seed,
        })
        print(f"  {name}: {fit_seconds:.1f}s fit, iterations={iterations}", flush=True)

    artifacts["ensemble"] = MeanEnsemble(
        tuple(artifacts[name] for name in ENSEMBLE_MEMBERS)
    )
    predictions["ensemble"] = {
        set_name: np.mean(
            [predictions[name][set_name] for name in ENSEMBLE_MEMBERS], axis=0
        )
        for set_name in prediction_sets
    }
    metadata.append({
        "model": "ensemble",
        "strategy_candidate": True,
        "train_rows": len(train_target),
        "stop_rows": len(stop_target),
        "features": len(features),
        "fit_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "iterations": None,
        "members": list(ENSEMBLE_MEMBERS),
        "weights": [0.25] * len(ENSEMBLE_MEMBERS),
        "quick": quick,
        "seed": seed,
    })
    return artifacts, predictions, metadata
