"""
Use any tabular regressor with SkPM
===================================

SkPM handles the process-aware part of predictive process monitoring: loading
an event log, splitting complete cases, creating targets, extracting features,
and encoding prefixes. The result is an ordinary numeric table, so the final
estimator can come from scikit-learn or another library with the same
``fit(X, y)`` and ``predict(X)`` interface.

This guide builds that preprocessing once, then uses it with scikit-learn,
XGBoost, and TabPFN on the BPI Challenge 2020 Request for Payment log.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor

from skpm import case_ids
from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.model_selection import train_test_split
from skpm.sequence_encoding import Aggregation, Windowing

import warnings

warnings.filterwarnings("ignore")

# %%
# Load and split a real event log
# -------------------------------
#
# :class:`~skpm.event_logs.BPI20RequestForPayment` downloads the public log on
# first use and caches it locally. SkPM represents every event with the same
# three-level index: case identifier, timestamp, and event identifier.
dataset = BPI20RequestForPayment()
print(dataset)


# %%
# Split before fitting or extracting features. The ``"unbiased"`` strategy
# keeps cases intact, removes incomplete boundary cases, and separates train
# and test in time. Its published settings are bundled with this loader, so no
# dates need to be copied into the guide.
train, test = train_test_split(dataset, strategy="unbiased")

pd.DataFrame(
    {
        "events": [len(train), len(test)],
        "cases": [case_ids(train).nunique(), case_ids(test).nunique()],
    },
    index=["training", "test"],
)


# %%
# Choose one target per event
# ---------------------------
#
# We predict the hours remaining until each event's case finishes. The target
# uses future events, so it stays outside the feature pipeline and is created
# separately for the training and test cases.
y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

pd.concat([train[["activity"]], y_train], axis="columns").head()


# %%
# Build the process-aware half
# ----------------------------
#
# The feature union works at event level:
#
# * ``TimestampExtractor`` derives elapsed time and time since the last event.
# * ``OneHotEncoder`` turns the current activity into numeric indicators.
#
# ``Aggregation`` then summarizes every observed prefix. In particular, the
# cumulative mean of each activity indicator is its relative frequency in the
# prefix. Every input event still produces one output row—one prediction
# moment—but that row now describes everything observed so far.
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
        # ("prefix_encoding", Aggregation(method="mean")),
        ("prefix_encoding", Windowing(n=3)),
        (
            "model",
            RandomForestRegressor(
                random_state=0,
                n_jobs=-1,
            ),
        ),
    ]
)


# %%
# Start with scikit-learn
# -----------------------
#
# Fitting the pipeline learns the activity vocabulary, fits the prefix encoder,
# and trains the regressor using the training cases only. Slicing off the last
# step makes the boundary visible: this numeric table is exactly what any final
# estimator receives.
pipeline.fit(train, y_train)
pipeline[:-1].transform(train).head()

sklearn_predictions = pipeline.predict(test)
sklearn_mae = mean_absolute_error(y_test, sklearn_predictions)


# %%
# A process-specific baseline keeps the score honest. ``ActivityMeanRegressor``
# reads the raw activity column and predicts its mean training target, falling
# back to the global mean for an unseen activity.

baseline = DummyRegressor().fit(train, y_train)
baseline_predictions = baseline.predict(test)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print(
    "Baseline MAE: "
    f"{baseline_mae:.2f} hours\n"
    "RandomForestRegressor MAE: "
    f"{sklearn_mae:.2f} hours"
)

# %%
# Swap in XGBoost
# ----------------
#
# XGBoost exposes a scikit-learn regressor, so no SkPM adapter is necessary.
# Install the optional dependency first::
#
#     pip install "xgboost>=2.1.4"
#
# ``clone`` creates a fresh, unfitted copy of the complete pipeline. The named
# ``model`` step is the only part we replace; splitting, feature extraction,
# prefix encoding, fitting, and prediction all stay the same.
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
xgboost_predictions = xgboost_pipeline.predict(test)

print(
    "XGBoost MAE: "
    f"{mean_absolute_error(y_test, xgboost_predictions):.2f} hours"
)


# %%
# On a supported NVIDIA setup, adding ``device="cuda"`` to ``XGBRegressor``
# enables GPU training. Leaving it out, as above, gives a portable CPU example.


# %%
# Swap in TabPFN
# --------------
#
# TabPFN follows the same estimator interface. Install it separately because it
# brings its own model runtime and is not a core SkPM dependency::
#
#     pip install tabpfn
#
# Again, one line changes the estimator and the process-aware half remains
# untouched.
from tabpfn import TabPFNRegressor

tabpfn_pipeline = clone(pipeline).set_params(model=TabPFNRegressor())
tabpfn_pipeline.fit(train, y_train)
tabpfn_predictions = tabpfn_pipeline.predict(test)

print(
    "TabPFN MAE: "
    f"{mean_absolute_error(y_test, tabpfn_predictions):.2f} hours"
)


# %%
# TabPFN performs its own model-side preprocessing, so no scaler is needed.
# The one-hot step here has a process-specific purpose: ``Aggregation`` turns
# those indicators into activity frequencies over the prefix.
#
# The full BPI20 feature table contains tens of thousands of event samples. A
# GPU is therefore strongly recommended for this section; use ``device="auto"``
# (the default) or request ``device="cuda"`` explicitly. First use also
# downloads the TabPFN checkpoint and may require accepting its model license.
# For a small CPU experiment, subset **whole cases**, never individual events,
# so their prefixes and targets remain intact.


# %%
# The reusable boundary
# ---------------------
#
# SkPM owns everything that needs process semantics. The last pipeline step
# only sees rows of numbers and their targets. Any compatible tabular regressor
# can therefore replace ``model`` without rewriting the event-log workflow.

# %%
# Summary
# -------
#
# Using the configuration above, we obtained these held-out test results. Lower
# MAE is better.
#
# .. list-table:: Remaining-time results
#    :header-rows: 1
#    :widths: 3 1
#
#    * - Estimator
#      - MAE (hours)
#    * - Baseline
#      - 102.29
#    * - ``RandomForestRegressor``
#      - 97.60
#    * - ``XGBRegressor``
#      - 84.22
#    * - ``TabPFNRegressor``
#      - 80.15
#
# Treat these as reference values rather than guaranteed scores: dependency
# versions, hardware, and model settings can change the result.


# %%
# Run it in Google Colab
# ----------------------
#
# Prefer not to configure a local environment? Download the Jupyter notebook
# below, open `Google Colab <https://colab.research.google.com/>`_, choose
# **File → Upload notebook**, and select the downloaded ``.ipynb`` file. Add
# this as the first cell before running the guide::
#
#     %pip install "skpm @ git+https://github.com/raseidi/skpm.git"
#     %pip install "xgboost>=2.1.4" tabpfn
#
# Colab can provide a GPU runtime, which is strongly recommended for the TabPFN
# section. Choose **Runtime → Change runtime type → GPU** before fitting it.
