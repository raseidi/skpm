"""
Quickstart: the SkPM event log
==============================

SkPM extends scikit-learn to process mining. The one thing to understand first
is how it represents an event log — every transformer, target and split in the
package agrees on that shape.

This page builds a tiny log, converts it, and extracts a first set of features.
It uses synthetic data, so it runs anywhere in a second.
"""

# %%
# An event log is a table of events
# ---------------------------------
#
# Each row is one thing that happened: a **case** (the process instance), a
# **timestamp**, and an **activity**. Here, five purchase orders moving through
# a small fulfilment process.
#
# The column names below are XES names — the interchange format used across
# process mining — which SkPM recognizes by default. Named your columns
# differently? Pass ``column_mapping={"case_id": "...", ...}``, or set it once
# globally with :func:`skpm.config.set_global_config`.
import numpy as np
import pandas as pd

ACTIVITIES = ["receive order", "check stock", "pick items", "pack", "ship"]

rng = np.random.default_rng(7)
records = []
for case in range(1, 6):
    moment = pd.Timestamp("2024-03-04 08:00") + pd.Timedelta(days=case - 1)
    # Orders drop out at different stages, so cases have different lengths.
    for activity in ACTIVITIES[: rng.integers(3, len(ACTIVITIES) + 1)]:
        records.append(
            {
                "case:concept:name": f"Order-{case}",
                "time:timestamp": moment,
                "concept:name": activity,
            }
        )
        moment += pd.Timedelta(minutes=int(rng.integers(30, 600)))

raw = pd.DataFrame(records)
raw.head()


# %%
# ``to_event_log`` gives it SkPM's shape
# --------------------------------------
#
# :func:`skpm.to_event_log` is the single conversion boundary in SkPM. It sorts
# the log, numbers the events, and moves ``case_id`` and ``timestamp`` into a
# three-level index — ``(case_id, timestamp, event_id)``.
#
# Two consequences worth internalising:
#
# * **``activity`` stays a column; ``case_id`` and ``timestamp`` become keys.**
#   Feature extraction replaces the columns, so anything kept as a column is
#   consumed as data. Case and time are identity, not data, so they live in the
#   index and survive every step of a pipeline.
# * **``event_id`` makes each row addressable.** It is a plain 0-based counter
#   over the whole log, which keeps two events distinguishable even when a case
#   records them at the same timestamp.
from skpm import to_event_log

log = to_event_log(raw)
log


# %%
# Extracting features
# -------------------
#
# :class:`~skpm.feature_extraction.TimestampExtractor` derives numeric features
# from the timestamps. Features come at two levels, and the distinction is the
# useful part:
#
# * **case-level** features describe the event's position *in its own case* —
#   ``accumulated_time`` is how long the case has been running.
# * **event-level** features describe the moment itself — ``hour_of_day`` and
#   friends, normalized to roughly ±0.5 so they are model-ready (which is why
#   9 a.m. does not read as ``9`` below).
#
# Passing ``"all"`` selects every feature at a level; here we name a few to keep
# the table readable. Only **past-looking** features exist: nothing derived from
# future events, because that would leak the very thing you want to predict.
from skpm.feature_extraction import TimestampExtractor

features = TimestampExtractor(
    case_features=["accumulated_time", "time_since_last_event"],
    event_features=["hour_of_day", "day_of_week"],
    time_unit="h",
)
X = features.fit_transform(log)
X.round(2)


# %%
# Notice the index came through untouched. That is deliberate: every SkPM
# transformer preserves it, so downstream steps — and you, afterwards — can
# still tell which case and which moment a row belongs to.
#
# Plotting ``accumulated_time`` against each event's position in its case shows
# what a case-level feature means. Every line starts at zero and only ever
# rises: at any point it summarizes what has happened *so far*.
#
# :func:`skpm.trace_positions` reads that position off the index, one of a few
# accessors that save you from indexing into it by hand.
import matplotlib.pyplot as plt

from skpm import trace_positions

curves = X.assign(position=trace_positions(log))

fig, ax = plt.subplots(figsize=(6.5, 4))
for case, events in curves.groupby(level="case_id", sort=False):
    ax.plot(
        events["position"],
        events["accumulated_time"],
        marker="o",
        label=case,
    )

ax.set_xticks(range(int(curves["position"].max()) + 1))
ax.set_xlabel("position in case (0 = first event)")
ax.set_ylabel("accumulated time (hours)")
ax.set_title("A case-level feature, per case")
ax.legend(frameon=False, fontsize="small")
fig.tight_layout()


# %%
# Where to next
# -------------
#
# You now have the shape and a feature matrix. Two things stand between that
# and a prediction, and they get their own pages:
#
# * **Prefixes and targets** — an event log is not yet a supervised problem.
#   Each event is really a *prefix* of its case, and turning prefixes into fixed
#   -length rows is what :mod:`skpm.sequence_encoding` does. Labels come from
#   :mod:`skpm.feature_extraction.targets`, and the log has to be split before
#   any of it, with :func:`skpm.model_selection.train_test_split`.
# * **Selecting a model** — the same pattern on a real BPI Challenge log, with
#   cross-validation that keeps cases intact.
