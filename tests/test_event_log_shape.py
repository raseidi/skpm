"""Pipeline-invariant + canonical-shape tests for the event-log MultiIndex.

The whole package agrees on a data shape contract:

* ``case_id``, ``timestamp``, and ``event_id`` are the three index levels.
* ``event_id`` is a global per-event row counter assigned at load time.
* Inputs in XES (the default source naming) or the canonical names load
  directly; other namings are declared via ``set_global_config`` or
  ``column_mapping``. All resolve to the same canonical shape.
* Transformers preserve the MultiIndex across ``fit``/``transform`` so
  downstream steps can rely on it.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import (
    EVENT_LOG_INDEX,
    has_event_log_index,
    to_event_log,
)
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


def test_event_id_orders_events_within_case(flat_log):
    out = to_event_log(flat_log)
    # The full (case_id, timestamp, event_id) index is unique...
    assert out.index.is_unique
    # ...event_id is an integer level...
    assert pd.api.types.is_integer_dtype(out.index.get_level_values("event_id"))
    # ...and increases with event order within each case (holds whether
    # event_id is a global row counter or a per-case sequence).
    per_case = out.index.to_frame(index=False).groupby("case_id")["event_id"]
    assert per_case.apply(lambda s: s.is_monotonic_increasing).all()


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


def test_naming_styles_resolve_to_same_canonical_shape():
    """XES (default source naming), already-canonical, and declared-via-mapping
    inputs all land in identical canonical shape."""
    rows = {
        "case": [1, 1, 2],
        "act": ["a", "b", "a"],
        "ts": ["2024-01-01", "2024-01-02", "2024-01-01"],
    }
    xes = pd.DataFrame(
        {
            "case:concept:name": rows["case"],
            "concept:name": rows["act"],
            "time:timestamp": rows["ts"],
        }
    )
    canonical = pd.DataFrame(
        {
            "case_id": rows["case"],
            "activity": rows["act"],
            "timestamp": rows["ts"],
        }
    )
    title = pd.DataFrame(
        {
            "CaseID": rows["case"],
            "Activity": rows["act"],
            "Timestamp": rows["ts"],
        }
    )

    outs = [
        to_event_log(xes),  # XES = default source naming
        to_event_log(canonical),  # already canonical
        to_event_log(  # non-standard names declared explicitly
            title,
            column_mapping={
                "case_id": "CaseID",
                "activity": "Activity",
                "timestamp": "Timestamp",
            },
        ),
    ]
    for out in outs:
        assert tuple(out.index.names) == EVENT_LOG_INDEX
        assert list(out.columns) == ["activity"]
    # All three produce the same index and the same activity values.
    for out in outs[1:]:
        assert out.index.equals(outs[0].index)
        assert out["activity"].tolist() == outs[0]["activity"].tolist()


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
                ),
            ),
            ("agg", Aggregation(method="mean")),
        ]
    )
    out = pipe.fit_transform(canonical)
    assert isinstance(out, pd.DataFrame)
    assert tuple(out.index.names) == EVENT_LOG_INDEX
    assert len(out) == len(canonical)
