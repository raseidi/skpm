import numpy as np
import pandas as pd

import pytest
from skpm.feature_extraction.targets import (
    execution_time,
    next_activity,
    remaining_time,
)
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


def test_next_activity(dummy_data):
    out = next_activity(dummy_data)
    assert isinstance(out, pd.Series)
    assert len(out) == len(dummy_data)
    assert out.dtype == object
    assert out.name == "next_activity"
    assert tuple(out.index.names) == ("case_id", "timestamp", "event_id")


def test_remaining_time(dummy_data):
    out = remaining_time(dummy_data)
    assert isinstance(out, pd.Series)
    assert len(out) == len(dummy_data)
    assert out.dtype == float
    assert out.name == "remaining_time"
    assert tuple(out.index.names) == ("case_id", "timestamp", "event_id")


def test_execution_time(dummy_data):
    out = execution_time(dummy_data)
    assert isinstance(out, pd.Series)
    assert len(out) == len(dummy_data)
    assert out.dtype == float
    assert out.name == "execution_time"
    assert tuple(out.index.names) == ("case_id", "timestamp", "event_id")


def test_execution_time_values():
    # Case A: gaps of 30s then 60s; case B: one gap of 3600s.
    log = pd.DataFrame(
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
    out = execution_time(log)
    assert out.tolist() == [30.0, 60.0, 0.0, 3600.0, 0.0]
    assert execution_time(log, time_unit="h").tolist() == pytest.approx(
        [30 / 3600, 60 / 3600, 0.0, 1.0, 0.0]
    )
