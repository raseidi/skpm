"""Tests for the transformer contract in skpm.base."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer, CaseLevelTransformer
from skpm.event_logs.base import EVENT_LOG_INDEX, EventLog, to_event_log
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.case import VariantExtractor
from skpm.sequence_encoding import Bucketing, Indexing, Windowing


@pytest.fixture(name="log")
def fixture_log() -> pd.DataFrame:
    flat = pd.DataFrame(
        {
            "case:concept:name": np.repeat(np.arange(10), 5),
            "concept:name": np.random.choice(["a", "b", "c"], 50),
            "time:timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
        }
    )
    return to_event_log(flat)


# --- fitted marker (#4) -----------------------------------------------------


def test_fit_sets_fitted_marker(log):
    # Even transformers that set no other fitted attribute become introspectable
    # as fitted (previously the now-removed snapshot was their only marker).
    for est in (Bucketing(), Indexing()):
        est.fit(log)
        assert est.fitted_ is True
        check_is_fitted(est)  # no-arg form must not raise


def test_clone_is_unfitted(log):
    fitted = Bucketing().fit(log)
    assert hasattr(fitted, "fitted_")
    fresh = clone(fitted)
    assert not hasattr(fresh, "fitted_")
    with pytest.raises(Exception):
        check_is_fitted(fresh)


# --- centralized validation (#1) --------------------------------------------


@pytest.mark.parametrize(
    "est",
    [
        Bucketing(method="single"),
        Bucketing(method="prefix"),
        Indexing(),
        Windowing(n=2),
        TimestampExtractor(event_features="all"),
    ],
    ids=lambda e: f"{type(e).__name__}-{sorted(e.get_params().items())}",
)
def test_event_log_input_matches_dataframe_input(log, est):
    """An EventLog is accepted wherever a DataFrame is and gives an identical
    result — *including the index*.

    The index half is the subtle one: scikit-learn's ``set_output`` wrapper
    recovers an output index from the object passed to ``transform`` only when
    that object is itself a DataFrame (it deliberately avoids
    ``getattr(x, "index")``). So a transformer whose ``_transform`` returns a
    bare ndarray silently emits a flat index for EventLog callers. Every
    event-level transformer must therefore index its own output.
    """
    event_log = EventLog(dataframe=log.reset_index())

    via_frame = clone(est).fit(log).transform(log)
    via_object = clone(est).fit(event_log).transform(event_log)

    pd.testing.assert_frame_equal(
        pd.DataFrame(via_object), pd.DataFrame(via_frame)
    )
    assert tuple(via_object.index.names) == EVENT_LOG_INDEX


def test_event_log_fit_bookkeeping(log):
    event_log = EventLog(dataframe=log.reset_index())
    t = Indexing().fit(event_log)
    assert t.n_features_in_ == log.shape[1]
    assert list(t.feature_names_in_) == list(log.columns)
    assert len(Indexing().fit_transform(event_log)) == len(log)


def test_event_log_without_data_raises(log):
    with pytest.raises(ValueError, match="holds no event log"):
        Bucketing().fit(EventLog())


def test_non_log_input_still_raises_typeerror(log):
    # The message must name what *is* accepted, and point a single-column
    # (Series) caller at list slicing / ColumnTransformer.
    for bad in (log.values, log[log.columns[0]]):
        with pytest.raises(TypeError, match="EventLog or a pandas DataFrame"):
            Bucketing().fit(bad)
    with pytest.raises(TypeError, match="ColumnTransformer"):
        Bucketing().fit(log.values)


def test_transform_promotes_flat_input(log):
    # base.transform promotes a flat DataFrame so _transform always receives a
    # canonical event log: flat and canonical input yield identical results.
    # (Note: MultiIndex preservation on the *output* requires the input to
    # transform to already carry it — see test_pipeline_preserves_multiindex.)
    flat = log.reset_index()  # back to flat, canonical column names
    out_canonical = Bucketing(method="prefix").fit(log).transform(log)
    out_flat = Bucketing(method="prefix").fit(flat).transform(flat)
    assert (
        np.asarray(out_canonical).ravel().tolist()
        == np.asarray(out_flat).ravel().tolist()
    )


# --- output / feature-name check (#3) ---------------------------------------


class _MismatchedNames(BaseProcessTransformer):
    _parameter_constraints: dict = {}

    def get_feature_names_out(self):
        return ["declared"]

    def _transform(self, X, y=None):
        return pd.DataFrame({"actual": range(len(X))}, index=X.index)


def test_output_feature_name_mismatch_raises(log):
    t = _MismatchedNames().fit(log)
    with pytest.raises(ValueError, match="get_feature_names_out"):
        t.transform(log)


# --- cardinality split (#5) -------------------------------------------------


def test_cardinality_markers():
    assert Bucketing._cardinality == "event"
    assert Indexing._cardinality == "event"
    assert VariantExtractor._cardinality == "case"
    assert issubclass(VariantExtractor, CaseLevelTransformer)


def test_case_level_transformer_emits_one_row_per_case():
    # Variable-length traces (realistic): one row out per case.
    flat = pd.DataFrame(
        {
            "case:concept:name": [1, 1, 1, 2, 2, 3],
            "concept:name": ["a", "b", "c", "a", "b", "a"],
            "time:timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
        }
    )
    log = to_event_log(flat)
    n_cases = log.index.get_level_values("case_id").nunique()
    variants = VariantExtractor().fit(log).transform(log)
    assert len(variants) == n_cases
