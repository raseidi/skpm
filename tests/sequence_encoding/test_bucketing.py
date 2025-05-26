import numpy as np
import pandas as pd
import pytest
from skpm.sequence_encoding import Bucketing
from skpm.config import EventLogConfig as elc


@pytest.fixture(name="dummy_log")
def get_dummy_log():
    return pd.DataFrame(
        {
            elc.case_id: np.random.randint(1, 10, 100),
            elc.timestamp: pd.date_range("2021-01-01", periods=100, freq="6h"),
            elc.activity: np.random.choice(["a", "b", "c"], 100),
        }
    ).sort_values(elc.timestamp)


def test_single(dummy_log):
    bucketing = Bucketing(method="single")
    bucketing.fit(dummy_log)
    bucketing_values = bucketing.transform(dummy_log)
    assert isinstance(bucketing_values, pd.DataFrame)
    assert bucketing_values.shape == (len(dummy_log), 1)
    assert np.unique(bucketing_values) == "b1"

    bucketing = Bucketing().fit(dummy_log).transform(dummy_log)
    assert isinstance(bucketing, pd.DataFrame)


def test_prefix(dummy_log):
    bucketing = Bucketing(method="prefix")
    bucketing.fit(dummy_log)
    bucketing_values = bucketing.transform(dummy_log)
    assert isinstance(bucketing_values, pd.DataFrame)
    assert bucketing_values.shape == (len(dummy_log), 1)
    assert isinstance(len(np.unique(bucketing_values)), int)


def test_clustering_not_implemented(dummy_log):
    with pytest.raises(NotImplementedError):
        Bucketing(method="clustering").fit(dummy_log).transform(dummy_log)


def test_invalid_method(dummy_log):
    with pytest.raises(AssertionError):
        Bucketing(method="invalid_method").fit(dummy_log)


def test_output_feature_names():
    bucketing = Bucketing(method="single")
    feature_names = bucketing.get_feature_names_out()
    assert isinstance(feature_names, list)
    assert len(feature_names) == 1
    assert feature_names[0] == "bucket"
