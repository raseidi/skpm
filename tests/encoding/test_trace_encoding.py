import pytest
import numpy as np
import pandas as pd
import datetime as dt
from skpm.encoding import Aggregation, WindowAggregation
from skpm.config import EventLogConfig as elc

def test_aggregation():
    dummy_data = pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )

    # Test default aggregation
    rp = Aggregation().set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]

    # Test aggregation with different numerical method
    rp = Aggregation(num_method="sum").set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]

    # Test aggregation with invalid input data
    with pytest.raises(Exception) as exc_info:
        rp.transform(dummy_data[[elc.activity, elc.resource]])

def test_window():
    n = 100
    dummy_data = pd.DataFrame(
        {
            elc.case_id: np.random.choice(np.arange(0, 10), n),
            elc.activity + "_oh_1": np.random.randint(0, 2, n),
            elc.activity + "_oh_2": np.random.randint(0, 2, n),
            elc.resource + "_oh_1": np.random.randint(0, 3, n),
            elc.resource + "_oh_2": np.random.randint(0, 3, n),
            elc.resource + "_oh_3": np.random.randint(0, 3, n),
            "somehow_transformed_timestamp": np.random.rand(n),
        }
    )

    # Test windowed aggregation
    rp = WindowAggregation(num_method="sum", cat_method="sum").set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]

    # Test windowed aggregation with different window size and minimum events
    rp = WindowAggregation(window_size=3, min_events=2).set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]
