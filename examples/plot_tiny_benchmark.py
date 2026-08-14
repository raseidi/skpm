"""
==================================================
Select a remaining-time model for an event log
==================================================

This tutorial shows how SkPM and scikit-learn work together on a predictive
process monitoring problem. We will load a real event log, build a target,
compare regression models with cross-validation, and evaluate the selected
model on unseen cases.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from skpm import case_ids
from skpm.baselines import ActivityMeanRegressor
from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import TimestampExtractor
from skpm.model_selection import train_test_split
from skpm.feature_extraction.targets import remaining_time
from skpm.sequence_encoding import Aggregation


# %%
# Load a real event log
# ---------------------
#
# We use the BPI Challenge 2020 Request for Payment log. It records how payment
# requests moved through a Dutch university.
#
# ``BPI20RequestForPayment`` downloads the dataset the first time it is used
# and keeps a local copy for later runs. SkPM returns its standard event-log
# representation: case identifiers, timestamps, and event identifiers form the
# index, while attributes such as ``activity`` remain columns.
dataset = BPI20RequestForPayment()
print(dataset)

# %%
# Split before anything else
# --------------------------
#
# Splitting is the first step of a SkPM workflow — before feature extraction,
# before fitting anything. Fitting a transformer on the whole log and splitting
# afterwards leaks information from the test cases into training.
#
# An event log covers a limited period of time. Cases near the end may still be
# running when recording stops, which would make their durations incomplete.
# A random split can also train a model on later cases and test it on earlier
# ones. The ``"unbiased"`` strategy filters incomplete boundary cases and uses
# time order to create a more realistic test set.
#
# Its date and maximum-duration settings are constants published with each
# benchmark, so ``train_test_split`` reads them from the loader automatically.
# The split keeps every case entirely on one side, and returns two event logs.
train, test = train_test_split(dataset, strategy="unbiased")

split_summary = pd.DataFrame(
    {
        "events": [len(train), len(test)],
        "cases": [case_ids(train).nunique(), case_ids(test).nunique()],
    },
    index=["training", "test"],
)
split_summary


# %%
# Define the prediction target
# ----------------------------
#
# We will predict **remaining time**: the number of hours from the current
# event until its case finishes. In supervised machine learning this value is
# the target, usually called ``y``. It is calculated from future events, so it
# must stay separate from the input features to avoid target leakage.
#
# The split deliberately does not choose this for us: predicting remaining time
# is a regression, predicting the next activity a classification, and the same
# two logs serve either. So we build the target ourselves, once per side.
y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

pd.concat([train[["activity"]], y_train], axis="columns").head()


# %%
# Build a process-aware pipeline
# ------------------------------
#
# A scikit-learn ``Pipeline`` applies the same steps during cross-validation,
# training, and prediction:
#
# 1. ``TimestampExtractor`` creates two past-looking features: elapsed time
#    since the case started and time since its previous event. In parallel,
#    ``OneHotEncoder`` turns each activity label into numeric indicator columns.
#    This is necessary because labels are categories, not ordered numbers.
# 2. ``Aggregation`` summarizes the numeric features over the events observed
#    so far. For one-hot activity columns, their mean is the relative frequency
#    of each activity in the observed part of the case. This observed part is
#    called a **prefix**.
# 3. The regressor predicts one remaining-time value for each event.
#
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
                case_features=[
                    "accumulated_time",
                    "time_since_last_event",
                ],
                event_features=None,
                time_unit="h",
            ),
        ),
        ("activity", activity_features),
    ]
).set_output(transform="pandas")

# Pandas output preserves the event-log index that SkPM uses to identify cases.
pipeline = Pipeline(
    [
        ("features", features),
        ("prefix_encoding", Aggregation(method="mean")),
        ("model", LinearRegression()),
    ]
)


# %%
# Compare a baseline and regression models
# ----------------------------------------
#
# ``GridSearchCV`` fits every candidate on several training/validation splits.
# ``ActivityMeanRegressor`` is a simple process mining baseline: it predicts the
# average training-set remaining time for the current activity. A useful model
# should improve on this straightforward reference.
#
# Linear regression models a straight-line relationship, while a random forest
# and histogram gradient boosting can learn non-linear relationships. This
# small search keeps their SkPM preprocessing fixed. The baseline reads the raw
# activity column directly, so its grid entry skips the two preprocessing steps.
#
# ``GroupKFold`` again keeps cases intact. We score models with mean absolute
# error (MAE), the average absolute difference between the true and predicted
# remaining time. Lower MAE is better. 
candidate_models = [
    LinearRegression(),
    RandomForestRegressor(
        n_estimators=100,
        random_state=0,
        n_jobs=-1,
    ),
    HistGradientBoostingRegressor(
        loss="absolute_error",
        random_state=0,
        learning_rate=0.1,
    ),
]


def model_name(model):
    """Return a concise label for the comparison table."""
    if isinstance(model, ActivityMeanRegressor):
        return "Activity mean (baseline)"
    return type(model).__name__


search = GridSearchCV(
    pipeline,
    param_grid=[
        {"model": candidate_models},
        {
            "features": ["passthrough"],
            "prefix_encoding": ["passthrough"],
            "model": [ActivityMeanRegressor()],
        },
    ],
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=4),
)
search.fit(train, y_train, groups=case_ids(train))

cv_results = pd.DataFrame(
    {
        "model": [
            model_name(parameters["model"])
            for parameters in search.cv_results_["params"]
        ],
        "mean CV MAE (hours)": (-search.cv_results_["mean_test_score"]).round(
            2
        ),
    }
).sort_values("mean CV MAE (hours)")
cv_results.reset_index(drop=True)


# %%
# Evaluate the selected model
# ---------------------------
#
# The search refits the best pipeline on all training cases. We now use the
# untouched test cases once, giving a final estimate of how the pipeline
# performs on new process instances.
predictions = search.predict(test)
test_mae = mean_absolute_error(y_test, predictions)
selected_model = model_name(search.best_estimator_.named_steps["model"])

print(f"Selected model: {selected_model}")
print(f"Cross-validation MAE: {-search.best_score_:.2f} hours")
print(f"Test MAE: {test_mae:.2f} hours")

# %%
# This is the complete basic pattern: load an event log, split it, define a
# target on each side, compose SkPM transformers in a scikit-learn pipeline,
# and select a model with grouped cross-validation. On a real project, use the
# same pattern with a representative event log and an untouched test set.
#
# The same call works on a log of your own, with no loader class involved::
#
#     train, test = train_test_split(pd.read_csv("my_log.csv"))
#
# ``"temporal"`` is the default strategy because it needs no per-dataset
# constants. ``"unbiased"`` needs ``max_days`` — pass it explicitly for a log
# that does not ship it.
