# read version from installed package
from importlib.metadata import version

__version__ = version("skpm")


from sklearn import set_config

set_config(transform_output="pandas")

from skpm.event_logs.base import (
    LogLike,
    case_ids,
    event_ids,
    timestamps,
    to_event_log,
    trace_positions,
)
from skpm.model_selection import train_test_split

__all__ = [
    "__version__",
    "LogLike",
    "to_event_log",
    "train_test_split",
    "case_ids",
    "timestamps",
    "event_ids",
    "trace_positions",
]
