"""
End-to-end HPO benchmark for remaining-time prediction (in hours) on the BPI20
Request-for-Payment log.

A single ``GridSearchCV`` over a 3-stage pipeline compares:

* **feature extraction** — temporal features, progressively unioned with
  one-hot ``activity``, resource-role, and work-in-progress features;
* **encoding** — rolling aggregation (mean / sum / median) over a swept
  ``prefix_len`` window (``None`` = cumulative over the whole prefix per case),
  positional indexing over several lag counts ``n``, and aggregation augmented
  with a one-hot prefix **bucket** (whose inner ``prefix_len`` is swept too);
* **model** — LinearRegression, RandomForest, and HistGradientBoosting
  (sklearn's scalable gradient boosting, used so the full log fits the compute
  budget), with a small hyper-parameter sweep each.

Each encoder's own hyperparameters are tuned in-grid via the nested
``encoder__<param>`` syntax, crossed with every feature set and model.

The pipeline uses ``memory`` caching so feature/encoder transforms are computed
once per (feature, encoder, fold) and reused across the model sweep.

``activity`` is one-hot encoded, so an aggregation encoder turns it into prefix
activity-frequency features (the classic PPM aggregation encoding). The event
log keeps its canonical ``(case_id, timestamp, event_id)`` MultiIndex through
the whole pipeline, and cross-validation uses ``GroupKFold`` keyed on
``case_id`` so whole cases stay on one side of every split (no prefix leakage).

Run (full log, ~45 min — the prefix_len/n sweep widens the grid):
    uv run python examples/tiny_benchmark.py

Quick smoke run on a subsample:
    SKPM_BENCH_NCASES=250 uv run python examples/tiny_benchmark.py
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

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
#
# Each entry is a GridSearchCV sub-grid: it pins the encoder *object* on the
# pipeline's "encoder" step and sweeps that encoder's own hyperparameters via
# the nested ``encoder__<param>`` syntax. BPI20-RFP cases are short (median 5,
# p95 8 events), so the windows/lags stay small — a window of 10 would equal
# ``None`` (cumulative) for ~95% of cases.
#
# * Aggregation — sweep ``method`` and the rolling-window ``prefix_len``
#   (``None`` = cumulative aggregation over the whole prefix).
# * Indexing    — sweep the number of positional lags ``n``.
# * agg+bucket  — aggregation-mean unioned with a one-hot prefix bucket; its
#   inner aggregation's ``prefix_len`` is swept via ``encoder__agg__prefix_len``.

ENCODER_GRIDS = [
    {
        "encoder": [Aggregation()],
        "encoder__method": ["mean", "sum", "median"],
        "encoder__prefix_len": [None, 2, 3, 5],
    },
    {
        # resource_roles is an integer (categorical) column, so fill both gaps.
        "encoder": [Indexing(fill_num_value=0.0, fill_cat_value=-1)],
        "encoder__n": [2, 3, 5, 8],
    },
    {
        "encoder": [
            FeatureUnion([("agg", Aggregation(method="mean")), ("bucket", _bucket_ohe())])
        ],
        "encoder__agg__prefix_len": [None, 3],
    },
]

# --- models (consume the encoded feature vectors) ---------------------------

MODEL_GRIDS = [
    {"model": [LinearRegression()]},
    {
        "model": [RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)],
        "model__n_estimators": [100],
        "model__max_depth": [None, 16],
    },
    {
        # HistGradientBoosting: sklearn's scalable gradient boosting (the classic
        # GradientBoostingRegressor does not scale to the full log in budget).
        # Handles NaNs natively too.
        "model": [HistGradientBoostingRegressor(random_state=RANDOM_STATE)],
        "model__learning_rate": [0.05, 0.1],
    },
]

_FEATURE_LABEL = {id(v): k for k, v in FEATURE_TECHNIQUES.items()}


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
    """Return a *fresh* per-run cache dir, removed when the process exits.

    The pipeline's ``memory`` cache memoises each (feature, encoder, fold)
    transform so it is computed once and reused across the model sweep within a
    single ``search.fit``. It must NOT persist across runs: joblib keys the
    cache on the *unfitted* transformer's params + input data, never on skpm's
    source code. A cache written by an older skpm therefore hashes identically
    and is served as a stale hit, then deserialised into the current classes —
    yielding fitted transformers that lack attributes the current code expects
    (e.g. ``TimestampExtractor.case_features_``), which surfaces as a baffling
    ``AttributeError`` mid-fit. A unique temp dir per run sidesteps that while
    keeping the within-run reuse that makes the sweep affordable.
    """
    base = (
        os.path.join(os.environ["CLAUDE_JOB_DIR"], "tmp")
        if "CLAUDE_JOB_DIR" in os.environ
        else None
    )
    cache_dir = tempfile.mkdtemp(prefix="skpm_bench_cache_", dir=base)
    atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
    return cache_dir


def build_search() -> GridSearchCV:
    # memory caching => each (feature, encoder-config, fold) transform is
    # computed once and reused across every model in MODEL_GRIDS.
    pipe = Pipeline(
        [
            ("features", _timestamp()),
            ("encoder", Aggregation(method="mean")),
            ("model", LinearRegression()),
        ],
        memory=_cache_location(),
    )
    features = list(FEATURE_TECHNIQUES.values())

    param_grid = [
        # Per-activity mean baseline: passthrough feature/encoder steps let the
        # regressor read the raw activity label; compared under the same CV.
        {
            "features": ["passthrough"],
            "encoder": ["passthrough"],
            "model": [ActivityMeanRegressor()],
        },
    ]
    # Cross every encoder hyperparameter sub-grid with every model sub-grid,
    # over all feature techniques. Within each merged dict GridSearchCV takes
    # the full product of features x encoder-params x model-params.
    param_grid += [
        {"features": features, **enc, **mdl}
        for enc in ENCODER_GRIDS
        for mdl in MODEL_GRIDS
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
    """Readable label for a feature param value (handles 'passthrough')."""
    return mapping.get(id(obj), obj if isinstance(obj, str) else type(obj).__name__)


def _encoder_label(params: dict) -> str:
    """Readable encoder label for a candidate, including its swept hyperparams."""
    enc = params["encoder"]
    if isinstance(enc, str):  # "passthrough" (baseline)
        return enc
    if isinstance(enc, Aggregation):
        method = params.get("encoder__method", enc.method)
        prefix_len = params.get("encoder__prefix_len", enc.prefix_len)
        return f"agg-{method}(pl={prefix_len})"
    if isinstance(enc, Indexing):
        return f"index(n={params.get('encoder__n', enc.n)})"
    if isinstance(enc, FeatureUnion):  # agg-mean + prefix bucket
        prefix_len = params.get("encoder__agg__prefix_len")
        return f"agg-mean(pl={prefix_len})+bucket"
    return type(enc).__name__


def _model_params(params: dict) -> str:
    extra = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("model__")}
    return ", ".join(f"{k}={v}" for k, v in sorted(extra.items())) or "-"


def leaderboard(search: GridSearchCV, top: int = 12) -> pd.DataFrame:
    res = search.cv_results_
    rows = [
        {
            "features": _label(_FEATURE_LABEL, p["features"]),
            "encoder": _encoder_label(p),
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
    print(f"  encoder  : {_encoder_label(best)}")
    print(f"  model    : {type(best['model']).__name__} ({_model_params(best)})")
    print(f"  cv   MAE (h): {-search.best_score_:.2f}")
    print(f"  test MAE (h): {test_mae:.2f}")

    print("\n=== baselines (test MAE, h) ===")
    print(f"  global train mean       : {global_mean:.2f}")
    print(f"  per-activity train mean : {activity_baseline:.2f}")


if __name__ == "__main__":
    main()
