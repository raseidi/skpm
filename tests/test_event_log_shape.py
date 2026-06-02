"""Pipeline-invariant + canonical-shape tests for the event-log MultiIndex.

The whole package agrees on a data shape contract:

* ``case_id``, ``timestamp``, and ``event_id`` are the three index levels.
* ``event_id`` is a per-case sequence number assigned at load time.
* Transformers preserve the MultiIndex across ``fit``/``transform`` so
  downstream steps can rely on it.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import EVENT_LOG_INDEX, has_event_log_index, to_event_log
from skpm.feature_extraction import TimestampExtractor
from skpm.sequence_encoding import Aggregation


@pytest.fixture(name="flat_log")
def fixture_flat_log() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(50), n // 50),
            elc.activity: rng.choice(["a", "b", "c"], n),
            elc.resource: rng.integers(0, 5, n),
            elc.timestamp: pd.date_range("2024-01-01", periods=n, freq="2h"),
        }
    )


def test_to_event_log_sets_canonical_index(flat_log):
    out = to_event_log(flat_log)
    assert tuple(out.index.names) == EVENT_LOG_INDEX
    assert has_event_log_index(out)


def test_to_event_log_assigns_event_id_per_case(flat_log):
    out = to_event_log(flat_log)
    per_case_event_ids = (
        out.index.to_frame(index=False).groupby("case_id")["event_id"]
    )
    # Every case starts at 0 and is monotonically increasing.
    assert (per_case_event_ids.min() == 0).all()
    assert per_case_event_ids.apply(lambda s: s.is_monotonic_increasing).all()


def test_event_id_breaks_simultaneous_timestamp_ties():
    """Two events with the same case_id and timestamp must still differ in
    the MultiIndex via ``event_id``."""
    tied = pd.DataFrame(
        {
            elc.case_id: [1, 1, 1],
            elc.activity: ["start", "complete", "log"],
            elc.timestamp: ["2024-01-01 10:00:00"] * 3,
        }
    )
    out = to_event_log(tied)
    assert out.index.is_unique
    assert list(out.index.get_level_values("event_id")) == [0, 1, 2]


def test_to_event_log_is_idempotent(flat_log):
    once = to_event_log(flat_log)
    twice = to_event_log(once)
    assert once is twice  # already-canonical input is returned unchanged


def test_pipeline_preserves_multiindex(flat_log):
    """Multi-step Pipeline keeps the canonical MultiIndex at every stage."""
    canonical = to_event_log(flat_log)

    pipe = Pipeline(
        [
            (
                "time_features",
                TimestampExtractor(
                    case_features=None,
                    event_features=["day_of_week", "hour_of_day"],
                    targets=None,
                ),
            ),
            ("agg", Aggregation(method="mean")),
        ]
    )
    out = pipe.fit_transform(canonical)
    assert isinstance(out, pd.DataFrame)
    assert tuple(out.index.names) == EVENT_LOG_INDEX
    assert len(out) == len(canonical)
