import pandas as pd

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import EventLog, has_event_log_index, to_event_log

EOT_TOKEN: str = "<EOT>"


def _as_canonical(log: "pd.DataFrame | EventLog") -> pd.DataFrame:
    if isinstance(log, EventLog):
        log = log.dataframe
    return log if has_event_log_index(log) else to_event_log(log)


def next_activity(log: pd.DataFrame | EventLog) -> pd.Series:
    """Next activity within each case, as an index-aligned Series.

    The last event of each case gets ``EOT_TOKEN``. The result is indexed by
    the canonical ``(case_id, timestamp, event_id)`` MultiIndex, so it aligns
    with the event log and can be passed straight to ``pipe.fit(X, y)``.

    Returns
    -------
    pandas.Series
        Next-activity label per event, named ``"next_activity"``.
    """
    log = _as_canonical(log)
    out = log.groupby(level="case_id", sort=False, observed=True)[elc.activity].shift(
        -1, fill_value=EOT_TOKEN
    )
    return out.rename("next_activity")


def remaining_time(log: pd.DataFrame | EventLog, time_unit: str = "s") -> pd.Series:
    """Remaining time until each case ends, as an index-aligned Series.

    Parameters
    ----------
    log : pandas.DataFrame or EventLog
        Event log (flat or canonical).
    time_unit : {"s", "m", "h", "d", "w"}, default="s"
        Unit for the returned durations.

    Returns
    -------
    pandas.Series
        Remaining time per event (float), named ``"remaining_time"``, indexed
        by the canonical MultiIndex so it aligns with the event log.
    """
    from skpm.feature_extraction.case.time import TimestampCaseLevel

    log = _as_canonical(log)
    timestamps = log.index.get_level_values("timestamp").to_series(
        index=log.index, name="timestamp"
    )
    return TimestampCaseLevel.remaining_time(timestamps, time_unit=time_unit).rename(
        "remaining_time"
    )
