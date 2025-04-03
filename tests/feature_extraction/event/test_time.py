import numpy as np
import pandas as pd
import datetime as dt

import pytest
from skpm.feature_extraction import TimestampExtractor
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
    
def test_time(dummy_data):
    # test TimeStampExtractor
    t = TimestampExtractor()
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == t._n_features_out
    assert isinstance(out, pd.DataFrame)

    t = TimestampExtractor(case_features="execution_time", event_features=None)
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == 1
    assert isinstance(out, pd.DataFrame)

    t = TimestampExtractor(case_features="execution_time", event_features=["month_of_year", "day_of_week"])
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
