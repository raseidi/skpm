import numpy as np
import pandas as pd
import pytest

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import to_event_log
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


def test_indexing(pd_df):
    rp = Indexing()
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    Indexing(fill_value=-1).fit_transform(pd_df)

    # fill_value=None keeps the structurally-missing cells as NaN.
    out_nan = Indexing(fill_value=None).fit_transform(pd_df)
    assert out_nan.isna().to_numpy().sum() > 0


def test_indexing_rejects_removed_params():
    """Indexing is absolute, full-width, and encodes all input columns."""
    for kwargs in (
        {"n": 2},
        {"mode": "relative"},
        {"attributes": "activity"},
        {"fill_cat_value": 0},
        {"fill_num_value": 0.0},
    ):
        with pytest.raises(TypeError):
            Indexing(**kwargs)


def test_indexing_default_is_nan_free(pd_df):
    """Default Indexing zero-pads case-start positions (model-ready)."""
    out = Indexing().fit_transform(pd_df)
    assert out.isna().to_numpy().sum() == 0


def test_indexing_has_no_future_leakage(abc_log):
    """pos_j is the case's j-th event, revealed only once the prefix reaches it.

    Width is the longest case (here 3), so every prefix is fully represented.
    The promoted log has ``activity`` as its only column, so all columns are
    encoded.
    """
    out = Indexing(fill_value="PAD").fit_transform(abc_log)
    cols = ["activity_pos_0", "activity_pos_1", "activity_pos_2"]
    assert list(out.columns) == cols
    assert out[cols].to_numpy().tolist() == [
        ["a", "PAD", "PAD"],  # prefix a..
        ["a", "b", "PAD"],  # prefix ab.
        ["a", "b", "c"],  # prefix abc
    ]


def test_indexing_multicase_numeric_no_leakage():
    """Absolute mode keeps cases isolated and pads short prefixes (no leakage).

    Two cases of unequal length with a numeric attribute: ``pos_j`` is the
    j-th event of *that* case, padded (``0.0``) until the prefix reaches it,
    and never carried across the case boundary. Select the ``num`` column on
    the canonical log so only it is encoded.
    """
    df = pd.DataFrame(
        {
            elc.case_id: ["p", "p", "p", "q", "q"],
            elc.activity: ["a", "b", "c", "d", "e"],
            "num": [10.0, 20.0, 30.0, 40.0, 50.0],
            elc.timestamp: pd.date_range("2020-01-01", periods=5, freq="h"),
        }
    )
    out = Indexing(fill_value=0.0).fit_transform(to_event_log(df)[["num"]])
    assert out.to_numpy().tolist() == [
        [10.0, 0.0, 0.0],  # p prefix 1
        [10.0, 20.0, 0.0],  # p prefix 2
        [10.0, 20.0, 30.0],  # p prefix 3
        [40.0, 0.0, 0.0],  # q prefix 1 (no spill-over from case p)
        [40.0, 50.0, 0.0],  # q prefix 2 (q has no 3rd event -> padded)
    ]
