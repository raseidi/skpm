"""
Encoding a prefix four ways
===========================

Each event in an event log stands for a **prefix** — everything observed in its
case up to and including that event. Prefixes have different lengths; tabular
models need a fixed-length vector. Closing that gap is what
:mod:`skpm.sequence_encoding` does.

This example runs the four encoders on the *same* five-event case, so the
difference between them is visible rather than described. See
:ref:`sequence_encoding` for the reference.
"""

# %%
# One case, five events
# ---------------------
#
# A single case is enough to see what each encoder produces. It loops back to
# ``pick items`` after a failed check, so order carries real information.
import pandas as pd

from skpm import to_event_log

raw = pd.DataFrame(
    {
        "case:concept:name": ["O-1"] * 5,
        "time:timestamp": pd.to_datetime(
            [
                "2024-03-04 08:00",
                "2024-03-04 09:30",
                "2024-03-04 14:00",
                "2024-03-05 10:00",
                "2024-03-05 16:30",
            ]
        ),
        "concept:name": [
            "receive",
            "pick items",
            "quality check",
            "pick items",
            "ship",
        ],
    }
)
log = to_event_log(raw)
log


# %%
# Encoders take numbers
# ---------------------
#
# ``activity`` is a string, and none of the encoders can summarise a string. So
# one-hot it first — this is the scikit-learn idiom, and inside a pipeline the
# preceding step usually emits numbers already.
#
# Each row below is one prediction moment: at row 2 the model knows the case has
# reached ``quality check`` and nothing beyond it.
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
X = one_hot.fit_transform(log[["activity"]])
X.columns = [c.removeprefix("activity_") for c in X.columns]
X


# %%
# ``Aggregation`` — order-agnostic
# --------------------------------
#
# A running statistic per column. The mean of a one-hot column is the **relative
# frequency** of that activity in the prefix, which is why ``pick items`` climbs
# to 0.40 by the last row: two of the five events so far.
#
# Fixed width, same column names, no notion of order.
from skpm.sequence_encoding import Aggregation

Aggregation(method="mean").fit_transform(X).round(2)


# %%
# ``Windowing`` — the recent past, in order
# -----------------------------------------
#
# ``_w_0`` is the current event, ``_w_1`` the previous one. The window slides
# with the prefix, so these are positions *relative* to now. The first event has
# no predecessor and its ``_w_1`` cells are padded with ``0``.
#
# Width is ``n`` times the number of input columns — it grows with the window,
# not with the case.
from skpm.sequence_encoding import Windowing

Windowing(n=2).fit_transform(X)


# %%
# ``Indexing`` — absolute positions
# ---------------------------------
#
# ``_pos_0`` is the case's first event, ``_pos_1`` its second — positions from
# the *start*, not from now. Each cell stays padded until the prefix reaches it,
# so no row sees an event beyond its own length: read the matrix as a staircase.
#
# The width is the longest case seen at fit, so the whole prefix stays visible.
# That is also the cost: on a log with long cases this is a wide matrix.
from skpm.sequence_encoding import Indexing

Indexing().fit_transform(X)


# %%
# ``Bucketing`` — grouping, not encoding
# --------------------------------------
#
# The odd one out: it labels a prefix instead of encoding it, so you can fit a
# separate model per group. ``method="prefix"`` buckets by trace position —
# predicting from a two-event prefix and from a twenty-event prefix are close to
# different problems.
#
# It reads the log directly, not the one-hot matrix, since it only needs
# position.
from skpm.sequence_encoding import Bucketing

Bucketing(method="prefix").fit_transform(log)


# %%
# Choosing one
# ------------
#
# On real data, measure. As a starting point:
#
# * :class:`~skpm.sequence_encoding.Aggregation` — you want *what* happened, not
#   *when*. Compact and the usual default; see
#   :ref:`sphx_glr_auto_examples_plot_quickstart.py`.
# * :class:`~skpm.sequence_encoding.Windowing` — recent history drives the
#   outcome, as in :ref:`sphx_glr_auto_examples_plot_next_activity.py`.
# * :class:`~skpm.sequence_encoding.Indexing` — position from the case start is
#   meaningful and cases are short.
# * :class:`~skpm.sequence_encoding.Bucketing` — you intend to fit one model per
#   prefix group.
#
# They are not exclusive. A :class:`~sklearn.pipeline.FeatureUnion` of
# ``Aggregation`` and ``Windowing`` gives a model both the summary and the
# recent order:
from sklearn.pipeline import FeatureUnion

FeatureUnion(
    [("summary", Aggregation(method="mean")), ("recent", Windowing(n=2))]
).set_output(transform="pandas").fit_transform(X).round(2)


# %%
# Encoding is optional
# --------------------
#
# Without any of these, scikit-learn treats every event as an independent
# sample described by its own features. That is a legitimate model and a good
# baseline — reach for an encoder when you want the model to see the path
# taken, not just the current step.
