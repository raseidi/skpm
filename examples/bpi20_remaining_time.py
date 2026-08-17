"""
Remaining time on a real log
============================

The complete workflow on the BPI Challenge 2020 Request for Payment log: load,
split without leaking, build a target, select a model with grouped
cross-validation, and swap in estimators from other libraries.

This page is **not executed** when the documentation is built — it downloads a
real log and fits several models. Download the notebook at the bottom to run
it, or see :ref:`sphx_glr_auto_examples_plot_quickstart.py` for the same
pattern on synthetic data.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from skpm import case_ids
from skpm.baselines import ActivityMeanRegressor
from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.model_selection import train_test_split
from skpm.sequence_encoding import Windowing

# %%
# Load
# ----
#
# :class:`~skpm.event_logs.BPI20RequestForPayment` records how payment requests
# moved through a Dutch university. It downloads on first use and caches
# locally.
dataset = BPI20RequestForPayment()
print(dataset)


# %%
# Split
# -----
#
# First, before extracting features or fitting anything.
#
# A log covers a limited recording window, and that window distorts both ends:
# cases already running when recording began are missing their start, and cases
# still running when it stopped have incomplete durations — which would make
# their remaining-time labels wrong. ``strategy="unbiased"`` drops those cases
# and separates train and test in time.
#
# Its dates and maximum duration are constants published with the benchmark.
# This loader ships them, so the call carries none. See :ref:`unbiased_split`.
train, test = train_test_split(dataset, strategy="unbiased")

pd.DataFrame(
    {
        "events": [len(train), len(test)],
        "cases": [case_ids(train).nunique(), case_ids(test).nunique()],
    },
    index=["training", "test"],
)


# %%
# Target
# ------
#
# Hours from each event until its case finishes. It is computed from future
# events, so it stays outside the feature pipeline and is built once per side.
# The split does not choose it for us: the same two logs would serve next
# activity, a classification. See :ref:`targets`.
y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

pd.concat([train[["activity"]], y_train.round(1)], axis="columns").head()


# %%
# The process-aware pipeline
# --------------------------
#
# Three steps, in this order:
#
# 1. :class:`~skpm.feature_extraction.TimestampExtractor` derives past-looking
#    temporal features, while :class:`~sklearn.compose.ColumnTransformer`
#    one-hot encodes the activity label — categories are not ordered numbers.
# 2. :class:`~skpm.sequence_encoding.Windowing` encodes each prefix as the
#    three most recent events, in order.
# 3. The regressor predicts one value per event.
#
# Every input event still produces exactly one output row — one prediction
# moment — but that row now describes what has been observed so far. See
# :ref:`composing`.
activity_features = ColumnTransformer(
    [
        (
            "one_hot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["activity"],
        )
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

features = FeatureUnion(
    [
        (
            "time",
            TimestampExtractor(
                case_features=["time_since_last_event"],
                event_features="all",
                time_unit="h",
            ),
        ),
        ("activity", activity_features),
    ]
).set_output(transform="pandas")

pipeline = Pipeline(
    [
        ("features", features),
        ("prefix", Windowing(n=3)),
        ("model", RandomForestRegressor(random_state=0, n_jobs=-1)),
    ]
)


# %%
# Slicing off the last step makes the boundary visible: this numeric table,
# plus ``y``, is exactly what the final estimator receives.
pipeline.fit(train, y_train)
pipeline[:-1].transform(train).head()


# %%
# Select a model
# --------------
#
# :class:`~sklearn.model_selection.GridSearchCV` fits every candidate on
# several training/validation splits.
# :class:`~sklearn.model_selection.GroupKFold` with the case identifiers as
# groups keeps cases intact — without it, a case's early events would train the
# model and its later events would validate it.
#
# :class:`~skpm.baselines.ActivityMeanRegressor` predicts the mean training
# target of each event's activity. It reads the raw ``activity`` column, so its
# grid entry disables the two preprocessing steps. A model that cannot beat it
# is not learning the process.
#
# Scoring is mean absolute error, in hours. Lower is better.
search = GridSearchCV(
    pipeline,
    param_grid=[
        {
            "model": [
                LinearRegression(),
                RandomForestRegressor(random_state=0, n_jobs=-1),
                HistGradientBoostingRegressor(
                    loss="absolute_error", random_state=0
                ),
            ]
        },
        {
            "features": ["passthrough"],
            "prefix": ["passthrough"],
            "model": [ActivityMeanRegressor()],
        },
    ],
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=4),
)
search.fit(train, y_train, groups=case_ids(train))

pd.DataFrame(
    {
        "model": [
            type(params["model"]).__name__
            for params in search.cv_results_["params"]
        ],
        "mean CV MAE (hours)": (
            -search.cv_results_["mean_test_score"]
        ).round(2),
    }
).sort_values("mean CV MAE (hours)").reset_index(drop=True)


# %%
# The prefix encoder is a hyper-parameter too. Adding
# ``"prefix": [Aggregation(method="mean"), Windowing(n=3), Windowing(n=5)]``
# to the grid compares encodings under the same cross-validation.


# %%
# Evaluate once
# -------------
#
# The search refits the best pipeline on all training cases. The untouched test
# cases are used once, giving a final estimate on new process instances.
predictions = search.predict(test)

print(f"Selected model:       {type(search.best_estimator_[-1]).__name__}")
print(f"Cross-validation MAE: {-search.best_score_:.2f} hours")
print(f"Test MAE:             {mean_absolute_error(y_test, predictions):.2f} hours")


# %%
# Swap in other estimators
# ------------------------
#
# The final step only sees rows of numbers, so any estimator with the
# scikit-learn interface fits there. :func:`~sklearn.base.clone` gives a fresh,
# unfitted copy of the whole pipeline; ``set_params`` replaces the named
# ``model`` step. Splitting, feature extraction and prefix encoding are
# untouched.
#
# XGBoost is an optional dependency::
#
#     pip install "xgboost>=2.1.4"
from xgboost import XGBRegressor

xgboost_pipeline = clone(pipeline).set_params(
    model=XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=0,
        n_jobs=-1,
    )
)
xgboost_pipeline.fit(train, y_train)

print(
    "XGBoost MAE: "
    f"{mean_absolute_error(y_test, xgboost_pipeline.predict(test)):.2f} hours"
)


# %%
# On a supported NVIDIA setup, adding ``device="cuda"`` to ``XGBRegressor``
# enables GPU training. Leaving it out keeps this example portable.
#
# TabPFN follows the same interface. It brings its own model runtime, so it is
# also installed separately::
#
#     pip install tabpfn
from tabpfn import TabPFNRegressor

tabpfn_pipeline = clone(pipeline).set_params(model=TabPFNRegressor())
tabpfn_pipeline.fit(train, y_train)

print(
    "TabPFN MAE: "
    f"{mean_absolute_error(y_test, tabpfn_pipeline.predict(test)):.2f} hours"
)


# %%
# TabPFN does its own model-side preprocessing, so no scaler is needed. The
# one-hot step stays because it is process-specific, not model-specific: it is
# what lets the prefix encoder work on activities at all.
#
# This feature table has tens of thousands of event samples, so a GPU is
# strongly recommended here. First use also downloads the TabPFN checkpoint.
# For a small CPU experiment, subset **whole cases**, never individual events,
# so their prefixes and targets stay intact.


# %%
# Reference results
# -----------------
#
# Measured with the pipeline above — ``TimestampExtractor`` plus one-hot
# activity, ``Windowing(n=3)``, and the stated estimator — against a
# predict-the-mean :class:`~sklearn.dummy.DummyRegressor`. Lower MAE is better.
#
# .. list-table:: Remaining-time results, held-out test cases
#    :header-rows: 1
#    :widths: 3 1
#
#    * - Estimator
#      - MAE (hours)
#    * - ``DummyRegressor`` (predict the mean)
#      - 102.29
#    * - ``RandomForestRegressor``
#      - 97.60
#    * - ``XGBRegressor``
#      - 84.22
#    * - ``TabPFNRegressor``
#      - 80.15
#
# Treat these as reference values, not guaranteed scores: dependency versions,
# hardware and model settings all move them. ``ActivityMeanRegressor`` is not
# listed here because it is scored by the grid search above rather than
# separately — read its number off that table when you run the page.
baseline_mae = mean_absolute_error(
    y_test, DummyRegressor().fit(train, y_train).predict(test)
)
print(f"Predict-the-mean baseline: {baseline_mae:.2f} hours")


# %%
# Run it in Google Colab
# ----------------------
#
# Prefer not to configure a local environment? Download the notebook below,
# open `Google Colab <https://colab.research.google.com/>`_, choose
# **File → Upload notebook**, and add this as the first cell::
#
#     %pip install "skpm @ git+https://github.com/raseidi/skpm.git"
#     %pip install "xgboost>=2.1.4" tabpfn
#
# Choose **Runtime → Change runtime type → GPU** before fitting TabPFN.
