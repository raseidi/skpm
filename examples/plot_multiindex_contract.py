"""
One index, every event
======================

Features change as they move through a pipeline; an event's identity should
not. SkPM gives every event one address:
``(case_id, timestamp, event_id)``.
"""

# %%
# Three levels, three jobs
# ------------------------
#
# ``case_id`` keeps events in the right process instance, ``timestamp`` places
# them in time, and ``event_id`` keeps the address unique when events happen at
# the same instant.
import pandas as pd

from skpm import to_event_log, trace_positions

raw = pd.DataFrame(
    {
        "case_id": ["Order-2", "Order-1", "Order-1", "Order-1"],
        "timestamp": [
            "2024-03-05 09:00",
            "2024-03-04 10:00",
            "2024-03-04 10:00",
            "2024-03-04 11:30",
        ],
        "activity": ["receive", "receive", "check", "ship"],
    }
)

log = to_event_log(raw)
log


# %%
# The first two ``Order-1`` events share a timestamp, but their full addresses
# remain unique. ``event_id`` is a global identity; for the position inside a
# case, use :func:`skpm.trace_positions`.
addresses = log.index.to_frame(index=False).assign(
    trace_position=trace_positions(log).to_numpy()
)
addresses


# %%
# Alignment comes for free
# ------------------------
#
# SkPM features and targets preserve those addresses. They therefore meet at
# the correct event without extra ID columns or manual merges.
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import next_activity

X = TimestampExtractor(
    case_features=["accumulated_time"],
    event_features=["hour_of_day"],
    time_unit="h",
).fit_transform(log)
y = next_activity(log)

print(f"Same event addresses: {X.index.equals(y.index)}")
X.assign(activity=log["activity"], next_activity=y)[
    ["activity", "accumulated_time", "next_activity"]
]
