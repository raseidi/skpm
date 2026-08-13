"""
Quickstart: temporal features from an event log
===============================================

Build a small synthetic log and extract past-looking temporal features.
"""
# %%
# A synthetic log keeps the docs build fast and network-free.
import pandas as pd

log = pd.DataFrame({
    "case:concept:name": ["c1"] * 3 + ["c2"] * 2,
    "time:timestamp": pd.to_datetime([
        "2024-01-01 09:00", "2024-01-01 10:30", "2024-01-01 12:00",
        "2024-01-02 08:00", "2024-01-02 09:15",
    ]),
    "concept:name": ["A", "B", "C", "A", "C"],
})

# %%
# ``to_event_log`` is skpm's single coercion boundary.
from skpm import to_event_log

elog = to_event_log(log)
elog

# %%
# Extract features. The canonical MultiIndex survives the transform.
from skpm.feature_extraction import TimestampExtractor

X = TimestampExtractor().fit_transform(elog)
X
