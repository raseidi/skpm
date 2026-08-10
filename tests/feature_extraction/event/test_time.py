import numpy as np
import pandas as pd
import datetime as dt

import pytest
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.event.time import TimestampEventLevel
from skpm.config import EventLogConfig as elc


@pytest.fixture(name="dummy_data")
def fixture_dummy_pd():
    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.timestamp: pd.date_range(
                start="1/1/2020",
                periods=1000,
            ),
        }
    )


def test_time(dummy_data):
    # case_id and timestamp live in the event-log MultiIndex; the extractor
    # output contains only the requested features.
    t = TimestampExtractor()
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == t._n_features_out
    assert isinstance(out, pd.DataFrame)
    assert tuple(out.index.names) == ("case_id", "timestamp", "event_id")

    t = TimestampExtractor(case_features="execution_time", event_features=None)
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == 1
    assert isinstance(out, pd.DataFrame)

    t = TimestampExtractor(
        case_features="execution_time",
        event_features=["month_of_year", "day_of_week"],
    )
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == 1 + 2
    assert isinstance(out, pd.DataFrame)

    with pytest.raises(Exception):
        t = TimestampExtractor(case_features=None, event_features=None)
        t.fit(dummy_data)
        out = t.transform(dummy_data)

    dummy_data = pd.DataFrame(
        {
            elc.case_id: [1, 1, 1, 2, 2, 2],
            elc.timestamp: ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", ""],
        }
    )
    t = TimestampExtractor()
    with pytest.raises(Exception):
        t.fit(dummy_data[[elc.case_id, elc.timestamp]])


# 2024-01-07 is a Sunday.
@pytest.mark.parametrize(
    "ts, expected",
    [
        ("2024-01-07 00:00:00", -0.5),  # Sunday midnight, week start
        ("2024-01-08 06:00:00", (86400 + 6 * 3600) / 604800 - 0.5),  # Monday
        ("2024-01-10 12:00:00", 0.0),  # Wednesday noon, mid-week
        (
            "2024-01-12 18:30:15",
            (5 * 86400 + 18 * 3600 + 30 * 60 + 15) / 604800 - 0.5,
        ),  # Friday
        ("2024-01-13 23:59:59", 604799 / 604800 - 0.5),  # Saturday, week end
    ],
)
def test_secs_since_sunday_values(ts, expected):
    X = pd.Series(pd.to_datetime([ts]))
    out = TimestampEventLevel.secs_since_sunday(X)
    assert out.iloc[0] == pytest.approx(expected)


def test_secs_since_sunday_increases_through_week():
    # Same clock time on consecutive days must yield strictly increasing
    # values; the old implementation reset every midnight.
    X = pd.Series(pd.date_range("2024-01-07 09:30:00", periods=7, freq="D"))
    out = TimestampEventLevel.secs_since_sunday(X)
    assert out.is_monotonic_increasing
    assert out.nunique() == 7


@pytest.mark.parametrize(
    "method, ts, expected",
    [
        ("day_of_week", "2024-01-08", -0.5),  # Monday
        ("day_of_week", "2024-01-07", 0.5),  # Sunday
        ("hour_of_day", "2024-01-08 00:00:00", -0.5),
        ("hour_of_day", "2024-01-08 23:00:00", 0.5),
    ],
)
def test_event_feature_bounds(method, ts, expected):
    X = pd.Series(pd.to_datetime([ts]))
    out = getattr(TimestampEventLevel, method)(X)
    assert out.iloc[0] == pytest.approx(expected)
