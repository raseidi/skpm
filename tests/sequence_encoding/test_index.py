import warnings

import numpy as np
import pandas as pd
import pytest

from skpm.config import EventLogConfig as elc
from skpm.sequence_encoding import Indexing


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


def test_indexing(pd_df):
    rp = Indexing()
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    with pytest.raises(Exception):
        rp = Indexing(n=0)
        rp.fit(pd_df)

    rp = Indexing(n=2, fill_cat_value="TEST", fill_num_value=-1)
    rp.fit_transform(pd_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        rp = Indexing(n=None)
        rp.fit_transform(pd_df)

    rp = Indexing(n=None, attributes=elc.activity)
    rp.fit_transform(pd_df)


def test_indexing_default_is_nan_free(pd_df):
    """Default Indexing zero-pads case-start lags so the output is model-ready.

    The first ``n`` events of every case have no value at deeper positions
    (the lag reaches before the case start). With the default fills these
    cells must be padded, not left as NaN, so estimators that reject NaN
    (e.g. GradientBoostingRegressor) work out of the box.
    """
    out = Indexing(n=3).fit_transform(pd_df)
    assert out.isna().to_numpy().sum() == 0


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


def test_indexing_absolute_is_default_and_has_no_future_leakage(abc_log):
    """mode='absolute' (the default) indexes by position in the case.

    pos_j holds the case's j-th event, revealed only once the prefix has
    reached it: a prefix never sees events beyond its own length.
    """
    out = Indexing(
        n=3, attributes=elc.activity, fill_cat_value="PAD"
    ).fit_transform(abc_log)
    cols = ["activity_pos_0", "activity_pos_1", "activity_pos_2"]
    assert list(out.columns) == cols
    assert out[cols].to_numpy().tolist() == [
        ["a", "PAD", "PAD"],  # prefix a..
        ["a", "b", "PAD"],  # prefix ab.
        ["a", "b", "c"],  # prefix abc
    ]


def test_indexing_relative_is_a_sliding_window(abc_log):
    """mode='relative' lags by offset: pos_0 current, pos_1 previous, ..."""
    out = Indexing(
        n=3, attributes=elc.activity, fill_cat_value="PAD", mode="relative"
    ).fit_transform(abc_log)
    cols = ["activity_pos_0", "activity_pos_1", "activity_pos_2"]
    assert out[cols].to_numpy().tolist() == [
        ["a", "PAD", "PAD"],  # at a: current a, nothing before
        ["b", "a", "PAD"],  # at b: current b, prev a
        ["c", "b", "a"],  # at c: current c, prev b, prev2 a
    ]


def test_indexing_absolute_multicase_numeric_no_leakage():
    """Absolute mode keeps cases isolated and pads short prefixes (no leakage).

    Two cases of unequal length with a numeric attribute: ``pos_j`` is the
    j-th event of *that* case, padded (``0.0``) until the prefix reaches it,
    and never carried across the case boundary.
    """
    df = pd.DataFrame(
        {
            elc.case_id: ["p", "p", "p", "q", "q"],
            elc.activity: ["a", "b", "c", "d", "e"],
            "num": [10.0, 20.0, 30.0, 40.0, 50.0],
            elc.timestamp: pd.date_range("2020-01-01", periods=5, freq="h"),
        }
    )
    out = Indexing(n=3, attributes="num", mode="absolute").fit_transform(df)
    assert out.to_numpy().tolist() == [
        [10.0, 0.0, 0.0],  # p prefix 1
        [10.0, 20.0, 0.0],  # p prefix 2
        [10.0, 20.0, 30.0],  # p prefix 3
        [40.0, 0.0, 0.0],  # q prefix 1 (no spill-over from case p)
        [40.0, 50.0, 0.0],  # q prefix 2 (q has no 3rd event -> padded)
    ]


def test_indexing_invalid_mode_raises(pd_df):
    with pytest.raises(Exception):
        Indexing(mode="bogus").fit(pd_df)
