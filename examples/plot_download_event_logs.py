"""
Public event logs
=================

:mod:`skpm.event_logs` ships loaders for fourteen public process mining
datasets — the BPI Challenge logs and Sepsis. Pick one; SkPM handles the
download, the XES parsing, the column normalization, and the local cache.
"""

# %%
# Construct a loader, get a log
# -----------------------------
#
# That is the whole API. This small BPI Challenge log is fetched from the 4TU
# repository on first use and read from the cache afterwards.
from skpm.event_logs import BPI13OpenProblems

log = BPI13OpenProblems()
print(log)
print(f"Cached in: {log.cache_folder}")


# %%
# A loader *is* a log
# -------------------
#
# Pass it straight to any SkPM transformer, target or split — ``.dataframe`` is
# available but never required. A flat DataFrame of your own works in exactly
# the same places; see :ref:`log_like`.
from skpm.feature_extraction.targets import remaining_time
from skpm.model_selection import train_test_split

train, test = train_test_split(log, test_size=0.2)
y_train = remaining_time(train, time_unit="h")

log.dataframe[["activity"]].head()


# %%
# Published split parameters
# --------------------------
#
# The unbiased split needs per-dataset constants published with the benchmark,
# not derived from the data. Six of the fourteen loaders carry them, so
# ``strategy="unbiased"`` is reproducible without copying dates out of a paper::
#
#     from skpm.event_logs import BPI17
#
#     train, test = train_test_split(BPI17(), strategy="unbiased")
#
# ``BPI13OpenProblems`` is not one of the six, which is why the split above is
# temporal. See :ref:`unbiased_split`.


# %%
# Explore another process
# -----------------------
#
# Swap in :class:`~skpm.event_logs.BPI17`,
# :class:`~skpm.event_logs.BPI20RequestForPayment`, or
# :class:`~skpm.event_logs.Sepsis`. Every loader shares the same download,
# cache, and DataFrame interface.
#
# Already have a XES file? :func:`~skpm.event_logs.read_xes` reads it into a
# flat DataFrame for :func:`skpm.to_event_log`::
#
#     from skpm import to_event_log
#     from skpm.event_logs import read_xes
#
#     log = to_event_log(read_xes("my_log.xes", n_jobs=-1))
