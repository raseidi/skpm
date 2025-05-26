from skpm.config import EventLogConfig as elc

class TimestampCaseLevel:
    """
    Extracts time-related features at the case level.

    Notes
    -----
    Separating the implementation for event-level and case-level features improves performance.
    The case-level implementation is slower due to the use of `groupby`.
    """

    TIME_UNIT_MULTIPLIER = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
    }

    @classmethod
    def accumulated_time(cls, case, ix_list, time_unit="s"):
        """Calculate the accumulated time from the start of each case in seconds."""
        return (
            case[elc.timestamp]
            .apply(lambda x: x - x.min())
            .loc[ix_list]
            .dt.total_seconds()
            / cls.TIME_UNIT_MULTIPLIER.get(time_unit, 1)
        )

    @classmethod
    def execution_time(cls, case, ix_list, time_unit="s"):
        """Calculate the execution time of each event in seconds.
        
        **NOTE**: This should be used as a target feature, since the _next_ step is 
        needed to calculate the execution time of each event."""
        return (
            case[elc.timestamp]
            .diff(-1)
            .dt.total_seconds()
            .fillna(0)
            .loc[ix_list]
            .abs() # to avoid negative numbers caused by diff-1
            / cls.TIME_UNIT_MULTIPLIER.get(time_unit, 1)
        )

    @classmethod
    def remaining_time(cls, case, ix_list, time_unit="s"):
        """Calculate the remaining time until the end of each case in seconds.
        
        **NOTE**: This should be used as a target feature, since the _last_ step 
        is needed to calculate the remaining time of each event."""
        
        return (
            case[elc.timestamp]
            .apply(lambda x: x.max() - x)
            .loc[ix_list]
            .dt.total_seconds()
            / cls.TIME_UNIT_MULTIPLIER.get(time_unit, 1)
        )