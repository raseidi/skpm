import numpy as np
import pandas as pd
import pytest
from skpm.bucketing import Bucketing
from skpm.config import EventLogConfig as elc


def get_dummy_log():
    return pd.DataFrame(
        {
            elc.case_id: np.random.randint(1, 10, 100),
            elc.timestamp: pd.date_range("2021-01-01", periods=100, freq="6h"),
            elc.activity: np.random.choice(["a", "b", "c"], 100),
        }
    ).sort_values(elc.timestamp)


def test_single():
    dummy_log = get_dummy_log()

    bucketing = Bucketing(method="single")
    bucketing.fit(dummy_log)
    bucketing_values = bucketing.transform(dummy_log)
    assert isinstance(bucketing_values, np.ndarray)
    assert bucketing_values.shape == (len(dummy_log),)
    assert np.unique(bucketing_values) == "b1"

    bucketing = (
        Bucketing().set_output(transform="pandas").fit(dummy_log).transform(dummy_log)
    )
    assert isinstance(bucketing, pd.DataFrame)


def test_others():
    dummy_log = get_dummy_log()

    bucketing = Bucketing(method="prefix")
    bucketing.fit(dummy_log)
    bucketing_values = bucketing.transform(dummy_log)
    assert isinstance(bucketing_values, np.ndarray)
    assert bucketing_values.shape == (len(dummy_log),)
    assert isinstance(len(np.unique(bucketing_values)), int)

    with pytest.raises(NotImplementedError) as exc_info:
        wip = Bucketing(method="clustering").fit(dummy_log).transform(dummy_log)
