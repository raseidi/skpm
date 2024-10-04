import pandas as pd
from skpm.config import EventLogConfig as elc
import numpy as np


def next_activity(log: pd.DataFrame):
    """Returns the next activity of each trace.

    Parameters
    ----------
    log : pd.DataFrame
        An event log.

    Returns
    -------
    pd.DataFrame
        A dataframe with the next activity of each trace.
    """
    return (
        log.groupby(elc.case_id, observed=True, as_index=True)[elc.activity]
        .shift(-1, fill_value="<EOT>")
        .values
    )


def remaining_time(log: pd.DataFrame, time_unit="seconds"):
    """Returns the remaining time of each trace.

    Parameters
    ----------
    log : pd.DataFrame
        An event log.

    Returns
    -------
    pd.DataFrame
        A dataframe with the remaining time of each trace.
    """
    if time_unit == "seconds":
        scaler = int(1e9)
    elif time_unit == "minutes":
        scaler = int(1e9 * 60)
    elif time_unit == "hours":
        scaler = int(1e9 * 60 * 60)
    elif time_unit == "days":
        scaler = int(1e9 * 60 * 60 * 24)
    else:
        raise ValueError(f"Time unit {time_unit} is not supported")
    return (
        log[[elc.case_id, elc.timestamp]]
        .groupby(elc.case_id, observed=True, as_index=False)[elc.timestamp]
        .apply(lambda x: (x.max() - x) / np.timedelta64(scaler, "ns"))
        # .reset_index(level=elc.case_id)
        .values
    )
