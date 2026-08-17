"""
Next activity prediction
========================

Remaining time is a regression; next activity is a classification. The log, the
split and the feature extraction are identical — only the target and the final
estimator change.

This example shows that swap, and why an order-preserving prefix encoding suits
this task.
"""

# %%
# The same process, the same split
# --------------------------------
#
# Orders that fail the quality check loop back to picking, so ``pick items``
# is sometimes followed by ``quality check`` and sometimes by ``pack``. Getting
# those cases right is the whole difficulty.
import numpy as np
import pandas as pd

from skpm import case_ids, to_event_log
from skpm.model_selection import train_test_split

rng = np.random.default_rng(11)
records = []
for case in range(400):
    moment = pd.Timestamp("2024-01-01") + pd.Timedelta(
        hours=float(rng.uniform(0, 24 * 180))
    )
    path = ["receive", "check stock", "pick items"]
    if rng.random() < 0.6:
        path.append("quality check")
        if rng.random() < 0.35:
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
        moment += pd.Timedelta(hours=float(rng.gamma(3.0, 2.0)))

log = to_event_log(pd.DataFrame(records))
train, test = train_test_split(log, test_size=0.25)
print(f"{len(train):,} training events in {case_ids(train).nunique():,} cases")


# %%
# The target
# ----------
#
# :func:`~skpm.feature_extraction.targets.next_activity` returns the activity
# that follows each event, with ``<EOT>`` marking the end of a case. Like every
# SkPM target it is a Series on the event-log index, so it aligns with the log
# without merging.
#
# ``<EOT>`` is a real class, not a missing value: predicting *that the case ends
# here* is part of the task.
from skpm.feature_extraction.targets import next_activity

y_train = next_activity(train)
y_test = next_activity(test)

pd.concat([train[["activity"]], y_train], axis="columns").head(8)


# %%
# Encoding the prefix in order
# ----------------------------
#
# Order carries more weight here than it does for remaining time. After
# ``[receive, check stock, pick items]`` and after
# ``[receive, check stock, pick items, quality check, pick items]`` the activity
# *frequencies* are similar, but the next activity is not the same.
#
# :class:`~skpm.sequence_encoding.Windowing` keeps the recent past in order —
# ``_w_0`` is the current event, ``_w_1`` the one before it. Applied after
# one-hot encoding, the model sees which activities occupied the last three
# steps and in which order. See :ref:`sequence_encoding`.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from skpm.feature_extraction import TimestampExtractor
from skpm.sequence_encoding import Aggregation, Windowing

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
        ("prefix", Windowing(n=3)),
        ("model", HistGradientBoostingClassifier(random_state=0)),
    ]
)
pipeline.fit(train, y_train)


# %%
# Does order help?
# ----------------
#
# Swapping the encoder is a one-line change, which makes the comparison cheap.
# The majority-class baseline is the reference any classifier has to beat.
#
# The ordered encoding wins, but by a couple of points rather than a landslide —
# most of the process is deterministic, so the two encodings only disagree on
# the branching events. Measure the swap on your own log rather than assuming
# either way.
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

scores = {}
for name, encoder in [
    ("Aggregation (order-agnostic)", Aggregation(method="mean")),
    ("Windowing, n=3 (ordered)", Windowing(n=3)),
]:
    candidate = clone(pipeline).set_params(prefix=encoder)
    candidate.fit(train, y_train)
    scores[name] = accuracy_score(y_test, candidate.predict(test))

baseline = DummyClassifier(strategy="most_frequent").fit(train, y_train)
scores["Most-frequent baseline"] = accuracy_score(
    y_test, baseline.predict(test)
)

pd.Series(scores, name="accuracy").sort_values(ascending=False).round(3)


# %%
# Where the errors are
# --------------------
#
# A single accuracy number hides which decisions are hard. The confusion matrix
# shows that most activities are near-deterministic — after ``pack`` comes
# ``ship`` — and that the difficulty is concentrated exactly where the process
# branches, on the events that may or may not loop back for a second check.
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pipeline.predict(test),
    normalize="true",
    values_format=".2f",
    colorbar=False,
    xticks_rotation=45,
    ax=ax,
)
ax.set_title("Next activity, held-out cases")
fig.tight_layout()


# %%
# Everything else is unchanged
# ----------------------------
#
# Same log, same split, same feature union. Only ``y`` and the final estimator
# differ from :ref:`sphx_glr_auto_examples_plot_quickstart.py` — which is the
# point: the process-aware half of the workflow is task-independent.
