"""scikit-learn estimator-contract conformance tests.

Two layers:

1. **sklearn's own data-free checks** — the subset of
   :mod:`sklearn.utils.estimator_checks` that validates the estimator API
   without feeding a plain 2D array (skpm estimators require an event-log
   DataFrame, so the array-based checks do not apply). These cover
   ``__init__`` not setting derived attributes, ``get_params``/``set_params``
   round-tripping, and default-constructibility.

2. **A domain contract test** — exercises ``fit``/``transform`` (or
   ``predict``) with a valid event log and asserts the invariants that the
   array-based checks would otherwise cover: param immutability across
   clone/fit, ``NotFittedError`` before fit, fit-time feature bookkeeping,
   feature-name/column agreement, and idempotent transform.

Together they are the regression net for the estimator-conformance fixes.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from skpm.baselines import ActivityMeanRegressor
from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import to_event_log
from skpm.feature_extraction import (
    ResourcePoolExtractor,
    TimestampExtractor,
    WorkInProgress,
)
from skpm.feature_extraction.case import VariantExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.sequence_encoding import Aggregation, Bucketing, Indexing


def _event_log(n: int = 300, n_cases: int = 30, seed: int = 0) -> pd.DataFrame:
    """A canonical event log with categorical and numeric feature columns.

    Cases have variable length (random ``case_id``), which keeps trace
    variants hashable for :class:`VariantExtractor`.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            elc.case_id: rng.integers(0, n_cases, n),
            elc.activity: rng.integers(0, 6, n).astype(str),
            elc.resource: rng.integers(0, 4, n).astype(str),
            "amount": rng.random(n),
            elc.timestamp: pd.date_range("2024-01-01", periods=n, freq="h"),
        }
    )
    return to_event_log(df)


def _all_estimators() -> list:
    """One default-constructed instance of every public estimator."""
    return [
        TimestampExtractor(),
        WorkInProgress(),
        ResourcePoolExtractor(),
        Aggregation(),
        Indexing(),
        Bucketing(),
        VariantExtractor(),
        ActivityMeanRegressor(),
    ]


SKLEARN_DATA_FREE_CHECKS = [
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_get_params_invariance,
    check_set_params,
]


@pytest.mark.parametrize(
    "estimator", _all_estimators(), ids=lambda e: type(e).__name__
)
@pytest.mark.parametrize(
    "check", SKLEARN_DATA_FREE_CHECKS, ids=lambda c: c.__name__
)
def test_sklearn_data_free_checks(check, estimator):
    """Run sklearn's array-free estimator checks on each estimator."""
    check(type(estimator).__name__, estimator)


def _contract_cases() -> list:
    log = _event_log()
    numeric_log = log[["amount"]].copy()  # canonical index preserved
    y = remaining_time(log, time_unit="h")
    return [
        pytest.param(
            TimestampExtractor(event_features="all", case_features=None),
            log,
            None,
            "transform",
            id="TimestampExtractor",
        ),
        pytest.param(
            WorkInProgress(), log, None, "transform", id="WorkInProgress"
        ),
        pytest.param(
            ResourcePoolExtractor(),
            log,
            None,
            "transform",
            id="ResourcePoolExtractor",
        ),
        pytest.param(
            Aggregation(method="mean"),
            numeric_log,
            None,
            "transform",
            id="Aggregation",
        ),
        pytest.param(Indexing(n=2), log, None, "transform", id="Indexing"),
        pytest.param(Bucketing(), log, None, "transform", id="Bucketing"),
        pytest.param(
            VariantExtractor(), log, None, "transform", id="VariantExtractor"
        ),
        pytest.param(
            ActivityMeanRegressor(),
            log,
            y,
            "predict",
            id="ActivityMeanRegressor",
        ),
    ]


@pytest.mark.parametrize("estimator,X,y,method", _contract_cases())
def test_estimator_contract(estimator, X, y, method):
    # 1. clone preserves the constructor params verbatim.
    params = estimator.get_params()
    assert clone(estimator).get_params() == params

    # 2. transform/predict before fit raises NotFittedError.
    with pytest.raises(NotFittedError):
        getattr(estimator, method)(X)

    # 3. fit returns self.
    fitted = estimator.fit(X) if y is None else estimator.fit(X, y)
    assert fitted is estimator

    # 4. fit does not mutate the constructor params.
    assert estimator.get_params() == params

    # 5. fit records sklearn's feature bookkeeping.
    assert isinstance(estimator.n_features_in_, int)
    assert estimator.n_features_in_ == X.shape[1]
    assert hasattr(estimator, "feature_names_in_")

    # 6. transform/predict is idempotent.
    out1 = getattr(estimator, method)(X)
    out2 = getattr(estimator, method)(X)
    if isinstance(out1, pd.DataFrame):
        pd.testing.assert_frame_equal(out1, out2)
    else:
        np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    # 7. declared feature names match the produced DataFrame columns.
    if hasattr(estimator, "get_feature_names_out") and isinstance(
        out1, pd.DataFrame
    ):
        assert list(out1.columns) == list(estimator.get_feature_names_out())
