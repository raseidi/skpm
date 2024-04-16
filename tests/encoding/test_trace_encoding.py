import polars as pl
import pytest
import numpy as np
import pandas as pd
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


def test_window_with_polars():
    n = 100
    pl_df = pl.DataFrame(
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
    rp = WindowAggregation(window_size=3, num_method="sum", cat_method="sum", engine="polars").set_output(
        transform="polars")
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
    assert isinstance(pl_agg, pd.DataFrame)
    assert pd_agg.equals(pl_agg)


def test_window_output():
    pl_df = pl.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )
    pd_df = pl_df.to_pandas()

    agg1 = WindowAggregation(window_size=3, num_method="sum", cat_method="sum", engine="pandas").set_output(
        transform="pandas")
    pd_agg = agg1.fit_transform(pd_df)
    agg2 = WindowAggregation(window_size=3, num_method="sum", cat_method="sum", engine="polars").set_output(
        transform="pandas")
    pl_agg = agg2.fit_transform(pl_df)
    print("pd:\n", pd_agg.head(15))
    print("pl:\n", pl_agg.head(15))
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
        agg1 = Aggregation(num_method="sum").set_output(transform="pandas")
        agg1.fit_transform(pl_df)

    with pytest.raises(ValueError):
        agg1 = WindowAggregation(num_method="sum").set_output(transform="pandas")
        agg1.fit_transform(pl_df)
