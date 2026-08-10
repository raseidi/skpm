import pandas as pd
import pytest

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import to_event_log
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.case.time import TimestampCaseLevel


@pytest.fixture(name="known_gaps_log")
def fixture_known_gaps_log():
    # Case a: gaps of 30s then 60s; case b: one gap of 3600s.
    return to_event_log(
        pd.DataFrame(
            {
                elc.case_id: ["a", "a", "a", "b", "b"],
                elc.activity: ["x", "y", "z", "x", "y"],
                elc.timestamp: pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:30",
                        "2024-01-01 00:01:30",
                        "2024-01-02 00:00:00",
                        "2024-01-02 01:00:00",
                    ]
                ),
            }
        )
    )


def _timestamps(log):
    return log.index.get_level_values("timestamp").to_series(index=log.index)


def test_time_since_last_event_values(known_gaps_log):
    out = TimestampCaseLevel.time_since_last_event(_timestamps(known_gaps_log))
    assert out.tolist() == [0.0, 30.0, 60.0, 0.0, 3600.0]
    assert out.index.equals(known_gaps_log.index)


def test_time_since_last_event_first_event_zero(known_gaps_log):
    out = TimestampCaseLevel.time_since_last_event(_timestamps(known_gaps_log))
    first_events = out.groupby(level="case_id", sort=False).head(1)
    assert (first_events == 0.0).all()


def test_time_since_last_event_time_unit_scaling(known_gaps_log):
    out = TimestampCaseLevel.time_since_last_event(
        _timestamps(known_gaps_log), time_unit="h"
    )
    assert out.tolist() == pytest.approx([0.0, 30 / 3600, 60 / 3600, 0.0, 1.0])

    t = TimestampExtractor(
        case_features="time_since_last_event",
        event_features=None,
        time_unit="h",
    )
    out = t.fit_transform(known_gaps_log)
    assert list(out.columns) == ["time_since_last_event"]
    assert out["time_since_last_event"].tolist() == pytest.approx(
        [0.0, 30 / 3600, 60 / 3600, 0.0, 1.0]
    )


def test_accumulated_time_values(known_gaps_log):
    out = TimestampCaseLevel.accumulated_time(_timestamps(known_gaps_log))
    assert out.tolist() == [0.0, 30.0, 90.0, 0.0, 3600.0]
