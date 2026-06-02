"""A tested 'happy path' quickstart: load -> features -> pipeline -> predict.

Keeps the documented entry-point workflow from rotting (runs on synthetic
data, no download).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

import skpm
from skpm.event_logs.base import to_event_log
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.sequence_encoding import Aggregation


def _synthetic_log(n_cases: int = 30, len_case: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_cases * len_case
    flat = pd.DataFrame(
        {
            "case:concept:name": np.repeat(np.arange(n_cases), len_case),
            "concept:name": rng.choice(["a", "b", "c"], n),
            "time:timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        }
    )
    return to_event_log(flat)


def test_quickstart_remaining_time_pipeline():
    log = _synthetic_log()

    # Target: index-aligned 1-D Series, kept separate from X (no leakage).
    y = remaining_time(log, time_unit="h")
    assert isinstance(y, pd.Series)
    assert y.ndim == 1
    assert tuple(y.index.names) == ("case_id", "timestamp", "event_id")

    pipe = Pipeline(
        [
            (
                "time",
                TimestampExtractor(
                    case_features=None,
                    event_features="all",
                    targets=None,
                    time_unit="h",
                ),
            ),
            ("encode", Aggregation(method="mean")),
            ("model", HistGradientBoostingRegressor(random_state=0)),
        ]
    )
    pipe.fit(log, y)
    preds = pipe.predict(log)
    assert len(preds) == len(log)


def test_public_accessors():
    log = _synthetic_log()
    assert skpm.case_ids(log).index.equals(log.index)
    assert skpm.timestamps(log).name == "timestamp"
    assert skpm.event_ids(log).nunique() == len(log)
    assert skpm.trace_positions(log).min() == 0
    # Accessors also accept a flat DataFrame (promoted on the fly).
    flat = log.reset_index()
    assert skpm.case_ids(flat).name == "case_id"
