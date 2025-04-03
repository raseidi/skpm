from skpm.config import EventLogConfig as elc

class TimestampEventLevel:
    """
    Provides methods to extract time-related features from the event level.

    Implementing event-level and case-level seperately makes code faster since here we do not need to group by case_id.

    """
    TIME_UNIT_MULTIPLIER = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
    }

    @classmethod
    def sec_of_min(cls, X):
        """Second of minute encoded as value between [-0.5, 0.5]"""
        return X.dt.second / 59.0 - 0.5

    @classmethod
    def min_of_hour(cls, X):
        """Minute of hour encoded as value between [-0.5, 0.5]"""

        return X.dt.minute / 59.0 - 0.5

    @classmethod
    def hour_of_day(cls, X):
        """Hour of day encoded as value between [-0.5, 0.5]"""

        return X.dt.hour / 23.0 - 0.5

    @classmethod
    def day_of_week(cls, X):
        """Hour of day encoded as value between [-0.5, 0.5]"""

        return X.dt.dayofweek / 6.0 - 0.5

    @classmethod
    def day_of_month(cls, X):
        """Day of month encoded as value between [-0.5, 0.5]"""
        return (X.dt.day - 1) / 30.0 - 0.5
    
    @classmethod
    def day_of_year(cls, X):
        """Day of year encoded as value between [-0.5, 0.5]"""

        return (X.dt.dayofyear - 1) / 365.0 - 0.5

    @classmethod
    def week_of_year(cls, X):
        """Week of year encoded as value between [-0.5, 0.5]"""
        return (X.dt.isocalendar().week - 1) / 52.0 - 0.5

    @classmethod
    def month_of_year(cls, X):
        """Month of year encoded as value between [-0.5, 0.5]"""
        return (X.dt.month - 1) / 11.0 - 0.5

    @classmethod
    def secs_within_day(cls, X):
        """Extract the number of seconds elapsed within each day from the timestamps encoded as value between [-0.5, 0.5]."""
        return (
            (X.dt.hour * 3600 + X.dt.minute * 60 + X.dt.second) / 86400
        ) - 0.5
        
    @classmethod
    def secs_since_sunday(cls, X):
        """Extract the number of seconds elapsed since the last Sunday from the timestamps encoded as value between [-0.5, 0.5]."""
        return (
            (X.dt.hour * 3600 + X.dt.minute * 60 + X.dt.second) / 604800
        ) - 0.5
        
    @classmethod
    def numerical_timestamp(cls, X, time_unit="s"):
        """Numerical representation of the timestamp."""
        return X.astype("int64") // 10**9 / cls.TIME_UNIT_MULTIPLIER.get(time_unit, 1)
