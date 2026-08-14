"""
Bring your own event log
========================

CSV, Parquet, database—if pandas can read it, SkPM can model it. Name the case,
time, and activity columns once; :func:`skpm.to_event_log` handles the rest.
"""

# %%
# Start with the table you already have
# -------------------------------------
#
# This small frame stands in for ``pd.read_csv(...)``. Its rows are deliberately
# out of order, and two events happen at the same time.
import pandas as pd

raw = pd.DataFrame(
    {
        "Order": ["PO-2", "PO-1", "PO-1", "PO-1"],
        "Occurred": [
            "2024-02-01 11:00+01:00",
            "2024-02-01 09:00+01:00",
            "2024-02-01 10:00+01:00",
            "2024-02-01 10:00+01:00",
        ],
        "Step": ["ship", "receive", "approve", "release"],
        "Operator": ["Lee", "Sam", "Lee", "Lee"],
        "Amount": [80, 120, 120, 120],
    }
)
raw


# %%
# Name the essentials
# -------------------
#
# Mapping keys are SkPM meanings; values are your column names. ``resource`` is
# optional, and business columns such as ``Amount`` pass through unchanged.
from skpm import to_event_log

log = to_event_log(
    raw,
    column_mapping={
        "case_id": "Order",
        "timestamp": "Occurred",
        "activity": "Step",
        "resource": "Operator",
    },
)
log


# %%
# Convert once, use everywhere
# ----------------------------
#
# The explicit conversion boundary makes the result easy to inspect: timestamps
# are UTC, rows are sorted by case and time, tied events keep their source order,
# and ``event_id`` makes every row unique. The canonical log is then ready for
# every SkPM API.
#
# SkPM also recognizes the XES names ``case:concept:name``,
# ``time:timestamp``, ``concept:name``, and optional ``org:resource`` without a
# mapping. Calling ``to_event_log`` on an already-canonical log is a no-op.
print(f"Canonical index: {log.index.names}")
print(f"Safe to convert twice: {to_event_log(log) is log}")
