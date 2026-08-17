"""
Features only a process has
===========================

Timestamps and activity labels are not the whole log. Three transformers read
signals that exist because the data describes a *process*: how busy the system
was, who does what, and which paths cases actually take.

None of them can be derived from a single event in isolation, which is what
makes them worth a dedicated transformer. See :ref:`feature_extraction`.
"""

# %%
# A log with resources and arrival bursts
# ---------------------------------------
#
# Two things are built into this log deliberately. Orders arrive in bursts, so
# the system is sometimes congested and sometimes idle. And each activity is
# handled by a distinct group of people — clerks take orders in and ship them,
# pickers move stock, inspectors check quality — but the log records only
# individual names, never the groups.
import numpy as np
import pandas as pd

HANDLERS = {
    "receive": ["Ada", "Ben"],
    "check stock": ["Ada", "Ben"],
    "pick items": ["Cleo", "Dan", "Eve"],
    "quality check": ["Fay", "Gus"],
    "pack": ["Cleo", "Dan", "Eve"],
    "ship": ["Ada", "Ben"],
}

rng = np.random.default_rng(5)
records = []
for case in range(500):
    # Bursty arrivals: a busy day every fortnight, quiet in between.
    day = float(rng.integers(0, 60))
    if rng.random() < 0.45:
        day = float(rng.choice([7, 21, 35, 49])) + rng.normal(0, 0.4)
    moment = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)

    path = ["receive", "check stock", "pick items"]
    if rng.random() < 0.6:
        path.append("quality check")
    path += ["pack", "ship"]

    for stage in path:
        records.append(
            {
                "case:concept:name": f"O{case:04d}",
                "time:timestamp": moment,
                "concept:name": stage,
                "org:resource": rng.choice(HANDLERS[stage]),
            }
        )
        moment += pd.Timedelta(hours=float(rng.gamma(3.0, 2.0)))

from skpm import to_event_log

log = to_event_log(pd.DataFrame(records))
log.head()


# %%
# ``resource`` is the one optional field
# --------------------------------------
#
# ``case_id``, ``timestamp`` and ``activity`` are required; ``resource`` is
# optional and exists for the transformer below. The XES name ``org:resource``
# is recognised by default — see :ref:`column_naming` for other names.
log["resource"].value_counts()


# %%
# Work in progress: reading across cases
# --------------------------------------
#
# Every other feature in SkPM looks at one case at a time.
# :class:`~skpm.feature_extraction.WorkInProgress` looks across them: it counts
# how many distinct cases were active in each time window, and labels every
# event with the count for its own window.
#
# This is a genuine process signal. A case queued behind two hundred others
# behaves differently from the same case in a quiet week, and no per-case
# feature can see that.
from skpm.feature_extraction import WorkInProgress

wip = WorkInProgress(window_size="1D").fit_transform(log)
wip.head()


# %%
# Plotted against time, the arrival bursts are plainly visible — and so is the
# feature's value, since a model given only per-case features cannot distinguish
# the peaks from the troughs.
import matplotlib.pyplot as plt

from skpm import timestamps

daily = wip["wip"].groupby(timestamps(log).dt.floor("D")).max()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(daily.index, daily.values, alpha=0.3)
ax.plot(daily.index, daily.values, linewidth=1)
ax.set_ylabel("cases active")
ax.set_title("Work in progress, one-day windows")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()


# %%
# .. warning::
#
#    ``WorkInProgress`` is the one transformer that reads other cases. Fit it
#    on the training log only — inside the ``Pipeline``, never on the full log
#    beforehand — or a training event's count can reflect test-set cases.


# %%
# Resource pools: roles from behaviour
# ------------------------------------
#
# :class:`~skpm.feature_extraction.ResourcePoolExtractor` never sees the
# ``HANDLERS`` table above. It correlates each resource's activity profile with
# every other's and groups the ones that behave alike, so two people who perform
# the same mix of activities land in the same role even if they never appear in
# the same case.
from skpm.feature_extraction import ResourcePoolExtractor

roles = ResourcePoolExtractor(threshold=0.7).fit_transform(log)

pd.crosstab(log["resource"], roles["resource_roles"])


# %%
# The recovered groups match the three teams the log was generated from, without
# an org chart. On a real log the collapse is what makes the feature usable: a
# one-hot over hundreds of individual resources is mostly noise, and breaks the
# moment someone new joins.
#
# A resource unseen at fit time maps to ``UNK`` and raises a
# :class:`~skpm.warnings.ConceptDriftWarning` — people joining a process is
# normal, but worth knowing about rather than silently absorbing.


# %%
# Trace variants: the paths cases take
# ------------------------------------
#
# A *trace variant* is the ordered tuple of activities of a case. Cases sharing
# one followed the same path, and counting them is the quickest way to see how
# much structure a log really has.
#
# :class:`~skpm.feature_extraction.case.variant.VariantExtractor` is a
# **case-level** transformer: it emits one row per case, not per event.
from skpm.feature_extraction.case.variant import VariantExtractor

extractor = VariantExtractor()
variants = extractor.fit_transform(log)
variants.head()


# %%
# Two variants cover this log because it was generated from two paths. Real
# logs are far less tidy — a few common variants and a long tail of one-offs,
# which is usually the first thing worth knowing about a process.
#
# ``inverse_transform`` maps the integer codes back to activity tuples.
counts = variants["variant"].value_counts()

for code, n in counts.items():
    path = " → ".join(extractor.inverse_transform([code])[0])
    print(f"{n:4d} cases   {path}")


# %%
# .. note::
#
#    Because it emits one row per case, ``VariantExtractor`` does not align
#    with an event-level target, so it is a **terminal** step rather than a
#    pipeline intermediate. Use it for analysis, not as an ``X`` for an
#    event-level model.


# %%
# In a pipeline
# -------------
#
# The two event-level transformers slot into a ``FeatureUnion`` alongside the
# temporal features, exactly like any other step.
from sklearn.pipeline import FeatureUnion

from skpm.feature_extraction import TimestampExtractor

process_features = FeatureUnion(
    [
        ("time", TimestampExtractor(event_features=None, time_unit="h")),
        ("wip", WorkInProgress(window_size="1D")),
        ("roles", ResourcePoolExtractor(threshold=0.7)),
    ]
).set_output(transform="pandas")

process_features.fit_transform(log).head()
