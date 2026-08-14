"""
Prefixes and targets
====================

A feature matrix is not yet a machine-learning problem. Two things are missing:
something to predict, and a way to turn a partly-finished case into one row.

This page covers both, plus the step that has to come before either —
splitting — and ends with a fitted model. It follows on from the
:ref:`sphx_glr_auto_examples_plot_quickstart.py` page, and uses synthetic data
so it runs in seconds.
"""

# %%
# A log of purchase orders
# ------------------------
#
# Four hundred orders over six months, with two sources of variation the model
# is never told about and has to recover from the data:
#
# * *express* orders skip the quality check and move much faster;
# * some standard orders fail the quality check and loop back to picking.
import numpy as np
import pandas as pd


def make_log(n_cases: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for case in range(n_cases):
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
    return pd.DataFrame(records)


from skpm import case_ids, to_event_log

log = to_event_log(make_log())
print(f"{len(log):,} events in {case_ids(log).nunique():,} cases")
log.head()


# %%
# Every event is a prediction moment
# ----------------------------------
#
# In process mining you do not predict once per case — you predict at every
# event, using only what has happened up to and including it. That growing slice
# is the case's **prefix**.
#
# So a case with six events contributes six training rows: the prefix of length
# 1, of length 2, and so on. This is why SkPM keeps one row per event, and why
# the targets below are also one value per event.
first_case = log.index.get_level_values("case_id")[0]
log.xs(first_case, level="case_id")[["activity"]]


# %%
# Split before anything else
# --------------------------
#
# Do this first, before extracting a single feature.
# :func:`skpm.model_selection.train_test_split` keeps every case entirely on one
# side, so no case has some of its prefixes in training and others in test.
#
# The order matters because the steps that follow are *fitted*: an encoder learns
# its categories, a scaler learns its means. Fit any of them on the whole log and
# test information has reached the model before you ever call ``predict``.
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
# Two targets from the same log
# -----------------------------
#
# :mod:`skpm.feature_extraction.targets` builds labels. The two most common
# process mining tasks are a regression and a classification over identical
# input:
#
# * :func:`~skpm.feature_extraction.targets.remaining_time` — hours until the
#   case finishes, reaching zero at the last event.
# * :func:`~skpm.feature_extraction.targets.next_activity` — which activity comes
#   next, with ``<EOT>`` marking the end of the case.
#
# Both are computed from *future* events, which is precisely why they are targets
# and never features. Each is a Series on the event-log index, so it lines up
# with the log without any merging on your part.
#
# Note they are functions, not transformers: labels stay outside the pipeline
# that transforms ``X``, following scikit-learn's separation of the two.
from skpm.feature_extraction.targets import next_activity, remaining_time

y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

pd.concat(
    [
        train[["activity"]],
        remaining_time(train, time_unit="h").round(1),
        next_activity(train),
    ],
    axis="columns",
).xs(first_case, level="case_id")


# %%
# Encoding a prefix
# -----------------
#
# A model needs a fixed number of columns, but prefixes keep growing. Resolving
# that is what :mod:`skpm.sequence_encoding` is for, and there is more than one
# sensible answer.
#
# :class:`~skpm.sequence_encoding.Aggregation` summarizes the prefix and ignores
# order. Applied to a one-hot activity column, its mean becomes the *relative
# frequency* of each activity so far — a compact summary of the path taken.
from sklearn.preprocessing import OneHotEncoder

from skpm.sequence_encoding import Aggregation, Windowing

one_case = log.xs(first_case, level="case_id", drop_level=False)[["activity"]]

one_hot = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
encoded = one_hot.fit_transform(one_case)
encoded.columns = [c.removeprefix("activity_") for c in encoded.columns]

Aggregation(method="mean").fit_transform(encoded).round(2)


# %%
# :class:`~skpm.sequence_encoding.Windowing` keeps order instead, giving a
# fixed-width view of the most recent events — here the current activity and the
# one before it. Positions falling outside the case are padded, so there are no
# missing values to clean up afterwards.
#
# (:class:`~skpm.sequence_encoding.Indexing` is the third option: one column per
# absolute position, keeping the whole prefix visible rather than a window of it.)
Windowing(n=2, fill_value="<none>").fit_transform(one_case)


# %%
# Putting it together
# -------------------
#
# The pipeline runs two branches in parallel and concatenates them.
#
# Only the activity branch is aggregated, and the reason is worth pausing on:
# ``accumulated_time`` is *already* a fact about the prefix, so averaging it
# again would replace "how long has this case been running" with "how long had it
# been running on average", discarding the model's most useful feature. Aggregate
# what needs summarizing; pass through what is already prefix-level.
#
# The activity column also has to be one-hot encoded *before* aggregation —
# averaging the string ``"pack"`` means nothing, and ``Aggregation`` says so
# rather than dropping the column quietly.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import FeatureUnion, Pipeline

from skpm.feature_extraction import TimestampExtractor

path_summary = Pipeline(
    [
        (
            "one_hot",
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
        ("prefix", Aggregation(method="mean")),
    ]
)

features = FeatureUnion(
    [
        (
            "elapsed",
            TimestampExtractor(
                case_features=["accumulated_time", "time_since_last_event"],
                event_features=None,
                time_unit="h",
            ),
        ),
        ("path", path_summary),
    ]
).set_output(transform="pandas")

pipeline = Pipeline(
    [
        ("features", features),
        ("model", HistGradientBoostingRegressor(random_state=0)),
    ]
)
pipeline.fit(train, y_train)


# %%
# Fitting used the training cases only, so scoring on the held-out cases is an
# honest estimate. Comparing against a model that always predicts the training
# mean shows the features earned their place.
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
# Predictions sharpen as the prefix grows
# ---------------------------------------
#
# A single error number hides the shape of the problem, which is the most
# important thing to understand about this task.
#
# On the left, one held-out case — the one whose duration is closest to the
# median, chosen from the data rather than by how well it scores. There is one
# prediction per prefix, each using only the events to its left.
#
# On the right is why the left looks the way it does. Early on, a prefix of
# ``[receive, check stock]`` is identical for an express order and a slow one, so
# the model can only predict something in between and is often well off. Each
# further event resolves more of that ambiguity, and the error falls sharply.
# Early predictions being unreliable is a property of the task, not a flaw in
# this model.
#
# The last two positions are the exception, and reading them carelessly would
# mislead: only the minority of orders that looped through a second quality check
# run that long, so those points average over a handful of events rather than all
# of them.
import matplotlib.pyplot as plt

from skpm import trace_positions

durations = y_test.groupby(level="case_id", observed=True).max()
median_case = (durations - durations.median()).abs().idxmin()

one_case_view = pd.DataFrame(
    {
        "position": trace_positions(test),
        "actual": y_test,
        "predicted": predictions,
    }
).xs(median_case, level="case_id")

error_by_position = (
    pd.DataFrame(
        {
            "position": trace_positions(test),
            "error": (y_test - predictions).abs(),
        }
    )
    .groupby("position")["error"]
    .mean()
)

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4))

left.plot(
    one_case_view["position"],
    one_case_view["actual"],
    marker="o",
    label="actual",
)
left.plot(
    one_case_view["position"],
    one_case_view["predicted"],
    marker="s",
    linestyle="--",
    label="predicted",
)
left.set_xticks(one_case_view["position"])
left.set_xlabel("position in case (0 = first event)")
left.set_ylabel("remaining time (hours)")
left.set_title(f"One case ({median_case})")
left.legend(frameon=False)

right.plot(error_by_position.index, error_by_position.values, marker="o")
right.set_xticks(error_by_position.index)
right.set_ylim(bottom=0)
right.set_xlabel("position in case (0 = first event)")
right.set_ylabel("mean absolute error (hours)")
right.set_title("All held-out cases")

fig.tight_layout()


# %%
# Where to next
# -------------
#
# That is the whole supervised loop: split, label, encode prefixes, fit, score.
#
# What remains is choosing *which* model, on a real log rather than a synthetic
# one — which is what the remaining-time model selection page does, with
# cross-validation that keeps cases intact.
#
# Switching to the other task is a two-line change: pass ``next_activity(train)``
# as ``y`` and put a classifier at the end of the pipeline.
