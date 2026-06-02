"""
End-to-end HPO benchmark for remaining-time prediction (in hours) on the BPI20
Request-for-Payment log.

A single ``GridSearchCV`` over a 3-stage pipeline compares:

* **feature extraction** — temporal features, progressively unioned with
  one-hot ``activity``, resource-role, and work-in-progress features;
* **encoding** — rolling aggregation (mean / sum / median), position indexing,
  and aggregation augmented with a one-hot prefix **bucket**;
* **model** — LinearRegression, RandomForest, and HistGradientBoosting
  (sklearn's scalable gradient boosting, used so the full log fits the compute
  budget), with a small hyper-parameter sweep each.

The pipeline uses ``memory`` caching so feature/encoder transforms are computed
once per (feature, encoder, fold) and reused across the model sweep.

``activity`` is one-hot encoded, so an aggregation encoder turns it into prefix
activity-frequency features (the classic PPM aggregation encoding). The event
log keeps its canonical ``(case_id, timestamp, event_id)`` MultiIndex through
the whole pipeline, and cross-validation uses ``GroupKFold`` keyed on
``case_id`` so whole cases stay on one side of every split (no prefix leakage).

Run (full log, ~15 min):
    uv run python examples/tiny_benchmark.py

Quick smoke run on a subsample:
    SKPM_BENCH_NCASES=250 uv run python examples/tiny_benchmark.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

import skpm
from skpm.baselines import ActivityMeanRegressor
from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import (
    ResourcePoolExtractor,
    TimestampExtractor,
    WorkInProgress,
)
from skpm.feature_extraction.targets import remaining_time
from skpm.sequence_encoding import Aggregation, Bucketing, Indexing

RANDOM_STATE = 0
TIME_UNIT = "h"
# Full log by default; set SKPM_BENCH_NCASES for a quick subsampled run.
N_CASES = int(os.environ["SKPM_BENCH_NCASES"]) if "SKPM_BENCH_NCASES" in os.environ else None


# --- building blocks --------------------------------------------------------


def _timestamp() -> TimestampExtractor:
    return TimestampExtractor(
        case_features=None, event_features="all", targets=None, time_unit=TIME_UNIT
    )


def _activity_ohe() -> ColumnTransformer:
    """One-hot the canonical ``activity`` column (drop the rest)."""
    return ColumnTransformer(
        [("activity", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["activity"])],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _bucket_ohe() -> Pipeline:
    """Prefix bucket label, one-hot encoded."""
    return Pipeline(
        [
            ("bucket", Bucketing(method="prefix")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


# --- feature-extraction techniques (consume the raw event log) --------------

FEATURE_TECHNIQUES = {
    "time": _timestamp(),
    "time+act": FeatureUnion([("time", _timestamp()), ("act", _activity_ohe())]),
    "time+act+res": FeatureUnion(
        [("time", _timestamp()), ("act", _activity_ohe()), ("res", ResourcePoolExtractor())]
    ),
    "time+act+res+wip": FeatureUnion(
        [
            ("time", _timestamp()),
            ("act", _activity_ohe()),
            ("res", ResourcePoolExtractor()),
            ("wip", WorkInProgress(window_size="2D")),
        ]
    ),
}

# --- encoding techniques (consume event-level features) ---------------------

ENCODING_TECHNIQUES = {
    "agg-mean": Aggregation(method="mean"),
    "agg-sum": Aggregation(method="sum"),
    "agg-median": Aggregation(method="median"),
    # resource_roles is an integer (categorical) column, so fill both gaps.
    "index-3": Indexing(n=3, fill_num_value=0.0, fill_cat_value=-1),
    "agg-mean+bucket": FeatureUnion(
        [("agg", Aggregation(method="mean")), ("bucket", _bucket_ohe())]
    ),
}

_FEATURE_LABEL = {id(v): k for k, v in FEATURE_TECHNIQUES.items()}
_ENCODER_LABEL = {id(v): k for k, v in ENCODING_TECHNIQUES.items()}


def load_log() -> pd.DataFrame:
    log = BPI20RequestForPayment(cache_folder="../ppm-llm/data/").dataframe
    if N_CASES is None:
        return log
    cases = log.index.get_level_values("case_id").unique()
    rng = np.random.default_rng(RANDOM_STATE)
    keep = rng.choice(cases, size=min(N_CASES, len(cases)), replace=False)
    return log[log.index.get_level_values("case_id").isin(keep)]


def case_holdout(log: pd.DataFrame, test_size: float = 0.25):
    """Case-level split (temporal split leaves an empty test set on this log)."""
    cases = log.index.get_level_values("case_id").unique()
    train_cases, test_cases = train_test_split(
        cases, test_size=test_size, random_state=RANDOM_STATE
    )
    level = log.index.get_level_values("case_id")
    return log[level.isin(train_cases)], log[level.isin(test_cases)]


def _cache_location() -> str:
    root = os.environ.get("CLAUDE_JOB_DIR")
    return os.path.join(root, "tmp", "bench_cache") if root else ".bench_cache"


def build_search() -> GridSearchCV:
    # memory caching => the feature/encoder transforms are computed once per
    # (feature, encoder, fold) and reused across the model sweep.
    pipe = Pipeline(
        [
            ("features", _timestamp()),
            ("encoder", Aggregation(method="mean")),
            ("model", LinearRegression()),
        ],
        memory=_cache_location(),
    )
    features = list(FEATURE_TECHNIQUES.values())
    encoders = list(ENCODING_TECHNIQUES.values())

    common = {"features": features, "encoder": encoders}
    param_grid = [
        # Per-activity mean baseline: passthrough feature/encoder steps let the
        # regressor read the raw activity label; compared under the same CV.
        {
            "features": ["passthrough"],
            "encoder": ["passthrough"],
            "model": [ActivityMeanRegressor()],
        },
        {**common, "model": [LinearRegression()]},
        {
            **common,
            "model": [RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)],
            "model__n_estimators": [100],
            "model__max_depth": [None, 16],
        },
        {
            # HistGradientBoosting: sklearn's scalable gradient boosting (the
            # classic GradientBoostingRegressor does not scale to the full log
            # in budget). Handles NaNs natively too.
            **common,
            "model": [HistGradientBoostingRegressor(random_state=RANDOM_STATE)],
            "model__learning_rate": [0.05, 0.1],
        },
    ]
    return GridSearchCV(
        pipe,
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=3),
        n_jobs=-1,
        error_score=np.nan,
        verbose=1,
    )


def _label(mapping: dict, obj) -> str:
    """Readable label for a feature/encoder param value (handles 'passthrough')."""
    return mapping.get(id(obj), obj if isinstance(obj, str) else type(obj).__name__)


def _model_params(params: dict) -> str:
    extra = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("model__")}
    return ", ".join(f"{k}={v}" for k, v in sorted(extra.items())) or "-"


def leaderboard(search: GridSearchCV, top: int = 12) -> pd.DataFrame:
    res = search.cv_results_
    rows = [
        {
            "features": _label(_FEATURE_LABEL, p["features"]),
            "encoder": _label(_ENCODER_LABEL, p["encoder"]),
            "model": type(p["model"]).__name__,
            "model_params": _model_params(p),
            "cv_mae_h": -m,
            "std": s,
        }
        for p, m, s in zip(res["params"], res["mean_test_score"], res["std_test_score"])
    ]
    df = pd.DataFrame(rows).sort_values("cv_mae_h", na_position="last")
    return df.head(top).reset_index(drop=True)


def main() -> None:
    log = load_log()
    train, test = case_holdout(log)
    y_train = remaining_time(train, time_unit=TIME_UNIT)
    y_test = remaining_time(test, time_unit=TIME_UNIT)
    groups = skpm.case_ids(train).to_numpy()

    print(
        f"cases: {log.index.get_level_values('case_id').nunique():,} | "
        f"train events: {len(train):,} | test events: {len(test):,}"
    )

    search = build_search()
    search.fit(train, y_train, groups=groups)

    board = leaderboard(search)
    n_failed = int(np.isnan(search.cv_results_["mean_test_score"]).sum())
    print("\n=== top configurations (3-fold GroupKFold, MAE in hours) ===")
    print(board.to_string(index=False))
    if n_failed:
        print(f"\n[warning] {n_failed} candidate(s) failed (NaN score)")

    best = search.best_params_
    test_mae = mean_absolute_error(y_test, search.predict(test))

    # --- baselines ---------------------------------------------------------
    global_mean = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
    # Per-activity mean: predict each event as the train mean remaining time of
    # its activity label (activities unseen in train fall back to global mean).
    activity_mean = y_train.groupby(train["activity"]).mean()
    pred_by_activity = test["activity"].map(activity_mean).fillna(y_train.mean())
    activity_baseline = mean_absolute_error(y_test, pred_by_activity)

    print("\n=== best pipeline ===")
    print(f"  features : {_label(_FEATURE_LABEL, best['features'])}")
    print(f"  encoder  : {_label(_ENCODER_LABEL, best['encoder'])}")
    print(f"  model    : {type(best['model']).__name__} ({_model_params(best)})")
    print(f"  cv   MAE (h): {-search.best_score_:.2f}")
    print(f"  test MAE (h): {test_mae:.2f}")

    print("\n=== baselines (test MAE, h) ===")
    print(f"  global train mean       : {global_mean:.2f}")
    print(f"  per-activity train mean : {activity_baseline:.2f}")


if __name__ == "__main__":
    main()
