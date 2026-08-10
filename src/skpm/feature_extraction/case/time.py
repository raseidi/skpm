import pandas as pd

from skpm.config import EventLogConfigMixin


class TimestampCaseLevel(EventLogConfigMixin):
    """Case-level time features.

    The methods take a ``pd.Series`` of timestamps indexed by the
    canonical event-log MultiIndex (``case_id`` is level 0) and return
    a Series aligned to the same index. They use ``groupby(level="case_id")``
    internally so they remain correct under any row order and benefit from
    pandas' fast grouped-aggregations.
    """

    TIME_UNIT_MULTIPLIER = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
    }

    @classmethod
    def accumulated_time(
        cls, timestamps: pd.Series, time_unit: str = "s"
    ) -> pd.Series:
        """Time elapsed since the first event of each case, in ``time_unit`` units."""
        first = timestamps.groupby(
            level=cls().case_id, sort=False, observed=True
        ).transform("min")
        return (
            timestamps - first
        ).dt.total_seconds() / cls.TIME_UNIT_MULTIPLIER.get(time_unit, 1)

    @classmethod
    def time_since_last_event(
        cls, timestamps: pd.Series, time_unit: str = "s"
    ) -> pd.Series:
        """Time since the previous event of the same case (0 for the first event), in ``time_unit`` units."""
        diffs = (
            timestamps.groupby(level=cls().case_id, sort=False, observed=True)
            .diff()
            .fillna(pd.Timedelta(0))
        )
        return diffs.dt.total_seconds() / cls.TIME_UNIT_MULTIPLIER.get(
            time_unit, 1
        )

    @classmethod
    def execution_time(
        cls, timestamps: pd.Series, time_unit: str = "s"
    ) -> pd.Series:
        """Time until the next event in the case (0 for the last event), in ``time_unit`` units."""
        diffs = (
            timestamps.groupby(level=cls().case_id, sort=False, observed=True)
            .diff(-1)
            .abs()
            .fillna(pd.Timedelta(0))
        )
        return diffs.dt.total_seconds() / cls.TIME_UNIT_MULTIPLIER.get(
            time_unit, 1
        )

    @classmethod
    def remaining_time(
        cls, timestamps: pd.Series, time_unit: str = "s"
    ) -> pd.Series:
        """Time remaining until the last event of each case, in ``time_unit`` units."""
        last = timestamps.groupby(
            level=cls().case_id, sort=False, observed=True
        ).transform("max")
        return (
            last - timestamps
        ).dt.total_seconds() / cls.TIME_UNIT_MULTIPLIER.get(time_unit, 1)
