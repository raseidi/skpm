import numpy as np
import pandas as pd

import pytest
from skpm.feature_extraction.targets import next_activity, remaining_time
from skpm.config import EventLogConfig as elc

@pytest.fixture(name="dummy_data")
def fixture_dummy_pd():
    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.timestamp: pd.date_range(
                start="1/1/2020", periods=1000,
            ),
        }
    )

def test_next_activity(dummy_data):
    # test next_activity
    out = next_activity(dummy_data)
    assert len(out) == len(dummy_data)
    assert isinstance(out, np.ndarray)
    assert out.dtype == object
    
def test_remaining_time(dummy_data):
    out = remaining_time(dummy_data)
    assert len(out) == len(dummy_data)
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
