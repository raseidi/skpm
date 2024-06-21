import polars as pl
import pytest
import numpy as np
import pandas as pd
from skpm.encoding import Aggregation
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

def test_aggregation_with_window():
    dummy_data = pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )

    # Test aggregation with different numerical method
    rp = Aggregation(window_size=3).set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]
        
    # Test window aggregation with window size larger than len(data) must work
    rp = Aggregation(window_size=len(dummy_data)+1).set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]
        
    # Test window aggregation with invalid window size
    with pytest.raises(Exception) as exc_info:
        rp = Aggregation(window_size=0).set_output(transform="pandas")
        rp.fit(dummy_data)
        out = rp.transform(dummy_data)


def test_aggregation_with_polars():
    pl_df = pl.DataFrame({
        elc.case_id: np.repeat(np.arange(0, 10), 100),
        elc.activity: np.random.randint(0, 10, 1000),
        elc.resource: np.random.randint(0, 3, 1000),
    })

    rp = Aggregation(engine="polars").set_output(transform="polars")
    rp.fit(pl_df)
    out = rp.transform(pl_df)
    assert isinstance(out, pl.DataFrame)
    assert out.height == pl_df.height

def test_aggregation_output():
    pl_df = pl.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )
    pd_df = pl_df.to_pandas()

    agg1 = Aggregation(num_method="sum").set_output(transform="pandas")
    pd_agg = agg1.fit_transform(pd_df)
    agg2 = Aggregation(num_method="sum", engine="polars").set_output(transform="pandas")
    pl_agg = agg2.fit_transform(pl_df)
    pl_agg = pl_agg.astype(pd_agg.dtypes)
    assert isinstance(pl_agg, pd.DataFrame)
    assert pd_agg.equals(pl_agg)


def test_wrong_dataframe_raises_exception():
    pl_df = pl.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )
    with pytest.raises(ValueError):
        agg1 = Aggregation(num_method="abc").set_output(transform="pandas")
        agg1.fit_transform(pl_df)
