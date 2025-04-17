import polars as pl
import pytest
import numpy as np
import pandas as pd
from skpm.sequence_encoding import Aggregation
from skpm.config import EventLogConfig as elc


@pytest.fixture(name="pd_df")
def fixture_dummy_pd():
    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )

def test_aggregation(pd_df):
    # Test default aggregation
    rp = Aggregation()
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    # Test aggregation with different numerical method
    rp = Aggregation(method="sum")
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    # Test aggregation with invalid input data
    with pytest.raises(Exception):
        rp.transform(pd_df[[elc.activity, elc.resource]])


def test_aggregation_with_window(pd_df):
    # Test aggregation with different numerical method
    rp = Aggregation(prefix_len=3)
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    # Test window aggregation with window size larger than len(data) must work
    rp = Aggregation(prefix_len=len(pd_df) + 1)
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    # Test window aggregation with invalid window size
    with pytest.raises(Exception):
        rp = Aggregation(prefix_len=0)
        rp.fit(pd_df)
        out = rp.transform(pd_df)


def test_aggregation_with_polars(pd_df):
    pl_df = pl.DataFrame(pd_df)

    rp = Aggregation(engine="polars")
    rp.fit(pl_df)
    out = rp.transform(pl_df)
    assert isinstance(out, pd.DataFrame)
    out = pl.DataFrame(out)
    assert out.height == pl_df.height


def test_aggregation_output(pd_df):
    pl_df = pl.DataFrame(pd_df)

    pd_agg = Aggregation(method="sum")
    pl_agg = Aggregation(method="sum", engine="polars")

    pd_agg = pd_agg.fit_transform(pd_df)
    pl_agg = pl_agg.fit_transform(pl_df)

    pd_agg = pd_agg.astype(pl_agg.dtypes)
    assert isinstance(pl_agg, pd.DataFrame)
    assert pd_agg.equals(pl_agg)

    pd_agg = Aggregation(prefix_len=3)
    pd_agg = pd_agg.fit_transform(pd_df)
    pl_agg = Aggregation(prefix_len=3, engine="polars")
    pl_agg = pl_agg.fit_transform(pl_df)
    pl_agg = pl_agg.astype(pd_agg.dtypes)
    assert isinstance(pl_agg, pd.DataFrame)
    assert pd_agg.equals(pl_agg)


def test_invalid_input(pd_df):
    # invalid arguments
    with pytest.raises(Exception):
        agg = Aggregation(method="abc")
        agg.fit_transform(pd_df)

    # invalid arguments
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        agg = Aggregation(engine="abc")
        agg.fit_transform(pd_df)

    # invalid input data
    with pytest.raises(AssertionError):
        agg = Aggregation()
        agg.fit(pd_df.values)

    # invalid input data
    with pytest.raises(AssertionError):
        agg = Aggregation().fit(pd_df)
        agg.transform(pd_df.values)

def test_methods(pd_df):
    methods = Aggregation._parameter_constraints["method"][0].options
    for method in methods:
        out_pd = Aggregation(method=method).fit_transform(pd_df)
        out_pl = Aggregation(method=method, engine="polars").fit_transform(pd_df)
        pd.testing.assert_frame_equal(out_pd, out_pl, check_dtype=False)
        
        # pandas engine
        assert isinstance(out_pd, pd.DataFrame)
        assert out_pd.shape[0] == pd_df.shape[0]
        
        # polars engine
        assert isinstance(out_pl, pd.DataFrame)
        assert out_pl.shape[0] == pd_df.shape[0]