"""
Download public event logs
==========================

SkPM provides ready-to-use loaders for 14 public process-mining datasets. Pick
a dataset; SkPM handles the download, parsing, normalization, and local cache.
"""

# %%
# Download once, reuse automatically
# ----------------------------------
#
# Constructing a loader is all it takes. This small BPI Challenge log is
# downloaded from 4TU on first use and loaded from its local cache afterwards.
from skpm.event_logs import BPI13OpenProblems

log = BPI13OpenProblems()
print(log)
print(f"Cached in: {log.cache_folder}")


# %%
# A ready-to-use event table
# --------------------------
#
# ``dataframe`` exposes ordinary pandas data with SkPM's standard
# ``(case_id, timestamp, event_id)`` index. The loader itself can also be passed
# directly to SkPM transformers, targets, and splitting functions.
events = log.dataframe
events[["activity"]].head()


# %%
# Explore another process
# -----------------------
#
# Swap in :class:`~skpm.event_logs.BPI17`,
# :class:`~skpm.event_logs.BPI20RequestForPayment`, or
# :class:`~skpm.event_logs.Sepsis`. Every loader in :mod:`skpm.event_logs` uses
# the same download, cache, and DataFrame interface.
