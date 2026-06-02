import polars as pl
import pytest
import numpy as np
import pandas as pd
from skpm.sequence_encoding import Aggregation
from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import to_event_log


@pytest.fixture(name="pd_df")
def fixture_dummy_pd():
    rng = np.random.default_rng(0)
    flat = pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: rng.integers(0, 10, 1000),
            elc.resource: rng.integers(0, 3, 1000),
            elc.timestamp: pd.date_range(
                "2024-01-01", periods=1000, freq="h"
            ),
        }
    )
    return to_event_log(flat)


@pytest.fixture(name="pd_df_flat")
def fixture_dummy_pd_flat(pd_df):
    """Flat form (no MultiIndex) sharing data with ``pd_df``."""
    return pd_df.reset_index()[["case_id", elc.activity, elc.resource]]

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

    # Test aggregation with invalid input data — flat DataFrame missing case_id
    with pytest.raises(Exception):
        rp.transform(pd_df.reset_index()[[elc.activity, elc.resource]])


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


@pytest.mark.skip(
    reason="polars engine temporarily disabled in _validate_log; re-enable with polars support"
)
def test_aggregation_with_polars(pd_df_flat):
    pl_df = pl.DataFrame(pd_df_flat)

    rp = Aggregation(engine="polars")
    rp.fit(pl_df)
    out = rp.transform(pl_df)
    assert isinstance(out, pd.DataFrame)
    out = pl.DataFrame(out)
    assert out.height == pl_df.height


@pytest.mark.skip(
    reason="polars engine temporarily disabled in _validate_log; re-enable with polars support"
)
def test_aggregation_output(pd_df, pd_df_flat):
    pl_df = pl.DataFrame(pd_df_flat)

    pd_agg = Aggregation(method="sum").fit_transform(pd_df)
    pl_agg = Aggregation(method="sum", engine="polars").fit_transform(pl_df)

    # The pandas engine preserves the event-log MultiIndex; flatten before
    # comparing values against the polars engine's flat output.
    pd_agg_flat = pd_agg.reset_index(drop=True)
    pd_agg_flat = pd_agg_flat.astype(pl_agg.dtypes)
    assert isinstance(pl_agg, pd.DataFrame)
    assert pd_agg_flat[pl_agg.columns].equals(pl_agg[pl_agg.columns])

    pd_agg = Aggregation(prefix_len=3).fit_transform(pd_df).reset_index(drop=True)
    pl_agg = Aggregation(prefix_len=3, engine="polars").fit_transform(pl_df)
    pl_agg = pl_agg.astype(pd_agg.dtypes)
    assert isinstance(pl_agg, pd.DataFrame)
    assert pd_agg[pl_agg.columns].equals(pl_agg[pl_agg.columns])


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

    # invalid input data — numpy array
    with pytest.raises(TypeError):
        Aggregation().fit(pd_df.values)

    with pytest.raises(TypeError):
        Aggregation().fit(pd_df).transform(pd_df.values)


def test_methods(pd_df):
    # pandas engine across all aggregation methods (polars cross-engine
    # comparison is suspended; see _validate_log).
    methods = Aggregation._parameter_constraints["method"][0].options
    for method in methods:
        out_pd = Aggregation(method=method).fit_transform(pd_df)
        assert isinstance(out_pd, pd.DataFrame)
        assert out_pd.shape[0] == pd_df.shape[0]
        assert tuple(out_pd.index.names) == ("case_id", "timestamp", "event_id")