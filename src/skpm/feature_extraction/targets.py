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
        .shift(-1, fill_value=elc.EOT)
        .values
    )


def remaining_time(log: pd.DataFrame, time_unit="s"):
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
    from skpm.feature_extraction import TimestampExtractor
    
    return TimestampExtractor(
        case_features=None, 
        event_features=None, 
        targets="remaining_time", 
        time_unit=time_unit
    ).set_output(transform="default").fit_transform(log)