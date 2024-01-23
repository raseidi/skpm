import pandas as pd
from skpm.config import EventLogConfig as elc


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


def remaining_time(log: pd.DataFrame):
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
    # TODO: this is implemented in skpm.event_feature_extraction.time.Timestamp. We should use that.
    return (
        log[[elc.case_id, elc.timestamp]]
        .groupby(elc.case_id, observed=True, as_index=False)[elc.timestamp]
        .apply(lambda x: x.max() - x)
        # .reset_index(level=elc.case_id)
        .values
    )
