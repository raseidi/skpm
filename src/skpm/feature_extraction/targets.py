import pandas as pd

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import has_event_log_index, to_event_log, EventLog

EOT_TOKEN: str = "<EOT>"


def next_activity(log: pd.DataFrame | EventLog) -> pd.Series:
    """Next activity within each case (``EOT_TOKEN`` for the last event)."""
    if not has_event_log_index(log):
        log = to_event_log(log)
    return (
        log.groupby(level="case_id", sort=False, observed=True)[elc.activity]
        .shift(-1, fill_value=EOT_TOKEN)
        .values
    )


def remaining_time(log: pd.DataFrame | EventLog, time_unit: str = "s") -> pd.Series:
    """Remaining time until the last event of each case."""
    from skpm.feature_extraction import TimestampExtractor

    out = (
        TimestampExtractor(
            case_features=None,
            event_features=None,
            targets="remaining_time",
            time_unit=time_unit,
        )
        .set_output(transform="default")
        .fit_transform(log)
    )
    return out["remaining_time"].values
