import numpy as np
import pandas as pd
import pytest
from skpm.event_feature_extraction import WorkInProgress
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
    with pytest.raises(TypeError):
        wip_empty_values = wip_empty.transform(empty_log)
        assert isinstance(wip_empty_values, np.ndarray)
        assert len(wip_empty_values) == 0
