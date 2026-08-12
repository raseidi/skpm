import numpy as np
import pandas as pd
import pytest
from skpm.feature_extraction.event import WorkInProgress
from skpm.config import EventLogConfig as elc


def test_wip():
    # Test with random data
    dummy_log = pd.DataFrame(
        {
            elc.case_id: np.random.randint(1, 10, 100),
            elc.timestamp: pd.date_range("2021-01-01", periods=100, freq="6h"),
            elc.activity: np.random.choice(["a", "b", "c"], 100),
        }
    ).sort_values(elc.timestamp)

    # Test fit_transform with default window_size
    wip = WorkInProgress()
    wip_values = wip.fit_transform(dummy_log)
    assert isinstance(wip_values, pd.DataFrame)
    assert wip_values.shape == (len(dummy_log), 1)

    # Test fit_transform with different window_size
    wip = WorkInProgress(window_size="2D")
    wip_values = wip.fit_transform(dummy_log)
    assert isinstance(wip_values, pd.DataFrame)
    assert wip_values.shape == (len(dummy_log), 1)

    # Test set_output with transform="pandas"
    wip_df = WorkInProgress().fit(dummy_log).transform(dummy_log)
    assert isinstance(wip_df, pd.DataFrame)

    # Test with empty dataframe
    empty_log = pd.DataFrame(columns=[elc.case_id, elc.timestamp, elc.activity])
    wip_empty = WorkInProgress()
    wip_empty.fit(empty_log)
    with pytest.raises(Exception):
        wip_empty.transform(empty_log)


def _log(rows):
    return pd.DataFrame(
        rows, columns=[elc.case_id, elc.timestamp, elc.activity]
    )


def test_wip_single_window():
    # A log fitting inside one window must not crash and every event gets
    # the window's active-case count.
    log = _log(
        [
            (1, "2024-01-01 08:00", "a"),
            (1, "2024-01-01 09:00", "b"),
            (2, "2024-01-01 10:00", "a"),
        ]
    )
    out = WorkInProgress(window_size="1D").fit_transform(log)
    assert out["wip"].tolist() == [2, 2, 2]


def test_wip_counts_active_cases_per_window():
    # WIP is the number of distinct cases active in each event's window --
    # including the last window, which must not be filled with an
    # unrelated event count.
    log = _log(
        [
            (1, "2024-01-01 08:00", "a"),
            (2, "2024-01-01 10:00", "a"),
            (2, "2024-01-02 10:00", "b"),
            (3, "2024-01-03 11:00", "a"),
            (4, "2024-01-03 12:00", "a"),
            (5, "2024-01-03 13:00", "a"),
        ]
    )
    out = WorkInProgress(window_size="1D").fit_transform(log)
    assert out["wip"].tolist() == [2, 2, 1, 3, 3, 3]


def test_wip_assigns_window_boundary_event_to_its_own_window():
    # An event exactly at a window start belongs to that window, not the
    # previous one.
    log = _log(
        [
            (1, "2024-01-01 23:00", "a"),
            (2, "2024-01-01 23:30", "a"),
            (3, "2024-01-02 00:00", "a"),
            (3, "2024-01-02 05:00", "b"),
        ]
    )
    out = WorkInProgress(window_size="1D").fit_transform(log)
    assert out["wip"].tolist() == [2, 2, 1, 1]


def test_wip_accepts_any_pandas_offset_alias():
    log = _log(
        [
            (1, "2024-01-01 08:00", "a"),
            (2, "2024-01-01 09:00", "a"),
            (2, "2024-01-01 13:00", "b"),
        ]
    )
    out = WorkInProgress(window_size="12h").fit_transform(log)
    assert out["wip"].tolist() == [2, 2, 1]

    # anchored (non-fixed) offsets work too
    weekly_log = _log(
        [
            (1, "2024-01-01 08:00", "a"),
            (2, "2024-01-03 08:00", "a"),
            (2, "2024-01-10 08:00", "b"),
        ]
    )
    out = WorkInProgress(window_size="W").fit_transform(weekly_log)
    assert out["wip"].tolist() == [2, 2, 1]


def test_wip_rejects_invalid_window_size():
    log = _log([(1, "2024-01-01 08:00", "a")])
    with pytest.raises(ValueError):
        WorkInProgress(window_size="not-an-alias").fit(log)
