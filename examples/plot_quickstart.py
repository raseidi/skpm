"""
Quickstart
==========

The whole SkPM workflow on one page: represent an event log, split it, build a
target, fit a pipeline, and score it against a baseline.

It runs on synthetic data in a couple of seconds. Each step links to the part
of the :ref:`user_guide` that explains it properly.
"""

# %%
# A log of purchase orders
# ------------------------
#
# An event log is a table of events: a **case** (the process instance), a
# **timestamp**, and an **activity**. Here, three hundred orders moving through
# a small fulfilment process, with two sources of variation the model is never
# told about — *express* orders skip the quality check and move faster, and
# some standard orders fail the check and loop back to picking.
#
# The column names are the XES names SkPM expects by default; see
# :ref:`column_naming` for logs named differently.
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
records = []
for case in range(300):
    moment = pd.Timestamp("2024-01-01") + pd.Timedelta(
        hours=float(rng.uniform(0, 24 * 180))
    )
    express = rng.random() < 0.4

    path = ["receive", "check stock", "pick items"]
    if not express:
        path.append("quality check")
        if rng.random() < 0.35:  # failed the check: pick again, re-check
            path += ["pick items", "quality check"]
    path += ["pack", "ship"]

    for stage in path:
        records.append(
            {
                "case:concept:name": f"O{case:04d}",
                "time:timestamp": moment,
                "concept:name": stage,
            }
        )
        moment += pd.Timedelta(
            hours=float(rng.gamma(3.0, 1.0 if express else 4.0))
        )

raw = pd.DataFrame(records)
raw.head()


# %%
# ``to_event_log`` gives it SkPM's shape
# --------------------------------------
#
# :func:`skpm.to_event_log` sorts the log, numbers the events, and moves
# ``case_id`` and ``timestamp`` into a three-level index. ``activity`` stays a
# column.
#
# That division is the one thing to internalise: columns are data, consumed and
# replaced by feature extraction; index levels are identity, and survive every
# step of a pipeline. See :ref:`event_log_shape`.
from skpm import case_ids, to_event_log

log = to_event_log(raw)
print(f"{len(log):,} events in {case_ids(log).nunique():,} cases")
log.head()


# %%
# Split first
# -----------
#
# Before extracting a single feature. :func:`skpm.model_selection.train_test_split`
# keeps every case entirely on one side, so no case has some of its events in
# training and others in test.
#
# The order matters because the steps that follow are *fitted*: an encoder
# learns its categories, a model learns everything. Fit any of them on the whole
# log and test information has reached the model before you call ``predict``.
# See :ref:`train_test_split`.
from skpm.model_selection import train_test_split

train, test = train_test_split(log, test_size=0.25)

pd.DataFrame(
    {
        "events": [len(train), len(test)],
        "cases": [case_ids(train).nunique(), case_ids(test).nunique()],
    },
    index=["training", "test"],
)


# %%
# Build the target
# ----------------
#
# The split deliberately returns two logs rather than four arrays: which target
# to predict is your decision. We predict **remaining time** — the hours until
# each event's case finishes.
#
# It is computed from future events, which is exactly why it is a target and
# never a feature, and why it is a plain function rather than a pipeline step.
# See :ref:`targets`.
from skpm.feature_extraction.targets import remaining_time

y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

pd.concat([train[["activity"]], y_train.round(1)], axis="columns").head()


# %%
# Fit a pipeline
# --------------
#
# One row per event, and each row describes the case's **prefix** — everything
# observed up to and including that event.
#
# :class:`~skpm.feature_extraction.TimestampExtractor` derives past-looking
# temporal features; the :class:`~sklearn.compose.ColumnTransformer` one-hot
# encodes the activity; :class:`~skpm.sequence_encoding.Aggregation` then
# summarises the prefix, turning those indicators into activity frequencies.
# See :ref:`composing`.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from skpm.feature_extraction import TimestampExtractor
from skpm.sequence_encoding import Aggregation

features = FeatureUnion(
    [
        (
            "time",
            TimestampExtractor(
                case_features=["accumulated_time", "time_since_last_event"],
                event_features=None,
                time_unit="h",
            ),
        ),
        (
            "activity",
            ColumnTransformer(
                [
                    (
                        "one_hot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                        ["activity"],
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            ),
        ),
    ]
).set_output(transform="pandas")

pipeline = Pipeline(
    [
        ("features", features),
        ("prefix", Aggregation(method="mean")),
        ("model", HistGradientBoostingRegressor(random_state=0)),
    ]
)
pipeline.fit(train, y_train)


# %%
# Score it
# --------
#
# Fitting used the training cases only, so scoring on the held-out cases is an
# honest estimate. Comparing against a model that always predicts the training
# mean shows whether the features earned their place.
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

predictions = pd.Series(pipeline.predict(test), index=test.index)

model_mae = mean_absolute_error(y_test, predictions)
baseline = DummyRegressor(strategy="mean").fit(train, y_train)
baseline_mae = mean_absolute_error(y_test, baseline.predict(test))

print(f"Predict-the-mean baseline: {baseline_mae:5.1f} hours MAE")
print(f"Model:                     {model_mae:5.1f} hours MAE")
print(f"Improvement:               {100 * (1 - model_mae / baseline_mae):4.0f}%")


# %%
# The index came through untouched
# --------------------------------
#
# Ten transformations later, every prediction still knows which case it belongs
# to and when it was made. That is what keeps post-hoc analysis cheap:
# :func:`skpm.trace_positions` reads how far into its case each event is, and
# grouping the error by it shows the shape of the problem.
#
# Early on, a prefix of ``[receive, check stock]`` is identical for an express
# order and a slow one, so the model can only predict something in between.
# Each further event resolves more of that ambiguity. Early predictions being
# unreliable is a property of the task, not a flaw in this model.
import matplotlib.pyplot as plt

from skpm import trace_positions

error_by_position = (
    (y_test - predictions).abs().groupby(trace_positions(test)).mean()
)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(error_by_position.index, error_by_position.values, marker="o")
ax.set_ylim(bottom=0)
ax.set_xticks(error_by_position.index)
ax.set_xlabel("position in case (0 = first event)")
ax.set_ylabel("mean absolute error (hours)")
ax.set_title("Predictions sharpen as the prefix grows")
fig.tight_layout()


# %%
# Where to next
# -------------
#
# That is the complete loop: represent, split, label, extract, fit, score.
#
# * :ref:`user_guide` — each step in detail.
# * :ref:`sphx_glr_auto_examples_plot_next_activity.py` — the same log as a
#   classification problem.
# * :ref:`sphx_glr_auto_examples_plot_prefix_encoding.py` — the other ways to
#   encode a prefix.
# * :ref:`sphx_glr_auto_examples_bpi20_remaining_time.py` — the same pattern on
#   a real BPI Challenge log, with model selection.
