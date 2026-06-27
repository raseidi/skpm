import numpy as np
import pandas as pd
import pytest

from skpm.config import EventLogConfig as elc
from skpm.sequence_encoding import Windowing


@pytest.fixture(name="abc_log")
def fixture_abc_log():
    """A single case with the deterministic sequence a -> b -> c."""
    return pd.DataFrame(
        {
            elc.case_id: ["k", "k", "k"],
            elc.activity: ["a", "b", "c"],
            elc.timestamp: pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                ]
            ),
        }
    )


@pytest.fixture(name="pd_df")
def fixture_dummy_pd():
    n = 1000
    start = pd.Timestamp("2000-01-01")
    end = pd.Timestamp("2024-01-01")
    timestamps = pd.to_datetime(
        np.linspace(start.value, end.value, n).astype(np.int64)
    )
    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), int(n / 10)),
            elc.activity: np.random.randint(0, 10, n).astype(str),
            elc.resource: np.random.rand(n),
            elc.timestamp: timestamps,
        }
    )


def test_windowing_is_a_sliding_window(abc_log):
    """w_0 is the current event, w_1 the previous one, etc."""
    out = Windowing(
        n=3, attributes=elc.activity, fill_cat_value="PAD"
    ).fit_transform(abc_log)
    cols = ["activity_w_0", "activity_w_1", "activity_w_2"]
    assert list(out.columns) == cols
    assert out[cols].to_numpy().tolist() == [
        ["a", "PAD", "PAD"],  # current a, nothing before
        ["b", "a", "PAD"],  # current b, prev a
        ["c", "b", "a"],  # current c, prev b, prev2 a
    ]


def test_windowing_default_n_columns(pd_df):
    """Default n=2 -> w_0 (current) and w_1 (previous) per attribute."""
    out = Windowing().fit_transform(pd_df)
    assert list(out.columns) == [
        "activity_w_0",
        "activity_w_1",
        "resource_w_0",
        "resource_w_1",
    ]
    assert out.shape[0] == pd_df.shape[0]


def test_windowing_default_is_nan_free(pd_df):
    """Default fills zero-pad the case-start positions (model-ready)."""
    out = Windowing(n=3).fit_transform(pd_df)
    assert out.isna().to_numpy().sum() == 0


def test_windowing_invalid_n_raises(pd_df):
    with pytest.raises(Exception):
        Windowing(n=0).fit(pd_df)
