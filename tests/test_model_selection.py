"""Tests for ``skpm.model_selection``.

The documented door for splitting an event log is
:func:`skpm.model_selection.train_test_split`. It returns **two event logs**
— never ``(X_train, X_test, y_train, y_test)``: choosing a target is a
modelling decision, made after the split, so the same split serves
remaining-time regression, next-activity classification, and unsupervised
log analysis alike.

Everything here is synthetic and network-free; the 4TU loaders are stood in
for by subclasses that skip the download path.
"""

import numpy as np
import pandas as pd
import pytest

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import (
    EVENT_LOG_INDEX,
    EventLog,
    TUEventLog,
    to_event_log,
)
from skpm.event_logs.split import temporal, unbiased
from skpm.model_selection import train_test_split

#: Published-looking parameters for the fake loader, shaped like the real ones.
FAKE_PARAMS = {"start_date": None, "end_date": None, "max_days": 5.0}


def _flat_log(n_cases: int = 40, events_per_case: int = 4) -> pd.DataFrame:
    """A flat log whose cases start on consecutive days and last ~2 hours."""
    rng = np.random.default_rng(0)
    rows = []
    for case in range(n_cases):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=case)
        for step in range(events_per_case):
            rows.append(
                {
                    elc.case_id: f"c{case:03d}",
                    elc.timestamp: start + pd.Timedelta(minutes=40 * step),
                    elc.activity: rng.choice(["a", "b", "c"]),
                }
            )
    return pd.DataFrame(rows)


class _FakeLoader(TUEventLog):
    """A 4TU-style loader that ships unbiased parameters, minus the download.

    ``TUEventLog.__init__`` downloads and parses; this bypasses it so the
    parameter-resolution behaviour can be tested offline.
    """

    _unbiased_split_params = dict(FAKE_PARAMS)

    def __init__(self, dataframe: pd.DataFrame):
        EventLog.__init__(self, dataframe=dataframe)


class _FakeLoaderNoParams(TUEventLog):
    """A loader that ships no unbiased parameters (8 of the 14 real ones)."""

    def __init__(self, dataframe: pd.DataFrame):
        EventLog.__init__(self, dataframe=dataframe)


@pytest.fixture(name="flat")
def fixture_flat() -> pd.DataFrame:
    return _flat_log()


@pytest.fixture(name="canonical")
def fixture_canonical(flat) -> pd.DataFrame:
    return to_event_log(flat)


@pytest.fixture(name="loader")
def fixture_loader(flat) -> _FakeLoader:
    return _FakeLoader(flat)


# --- return contract --------------------------------------------------------


def test_returns_exactly_two_event_logs(canonical):
    """The split returns (train, test) — never a 4-tuple carrying y."""
    out = train_test_split(canonical)

    assert isinstance(out, tuple)
    assert len(out) == 2
    train, test = out
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_preserves_the_canonical_multiindex(canonical):
    train, test = train_test_split(canonical)

    for side in (train, test):
        assert tuple(side.index.names) == EVENT_LOG_INDEX
    assert list(train.columns) == list(canonical.columns)


def test_whole_cases_stay_on_one_side(canonical):
    train, test = train_test_split(canonical)

    train_cases = set(train.index.get_level_values("case_id"))
    test_cases = set(test.index.get_level_values("case_id"))
    assert not train_cases & test_cases


# --- strategy dispatch ------------------------------------------------------


def test_defaults_to_the_temporal_strategy(canonical):
    """``temporal`` is the default: it needs no per-dataset parameters."""
    train, test = train_test_split(canonical)
    expected_train, expected_test = temporal(canonical)

    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


def test_temporal_strategy_forwards_test_size(canonical):
    train, test = train_test_split(
        canonical, strategy="temporal", test_size=0.4
    )
    expected_train, expected_test = temporal(canonical, test_size=0.4)

    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


def test_larger_test_size_yields_more_test_cases(canonical):
    _, small = train_test_split(canonical, test_size=0.2)
    _, large = train_test_split(canonical, test_size=0.5)

    assert (
        large.index.get_level_values("case_id").nunique()
        > small.index.get_level_values("case_id").nunique()
    )


def test_unbiased_strategy_matches_the_primitive(canonical):
    train, test = train_test_split(
        canonical, strategy="unbiased", **FAKE_PARAMS
    )
    expected_train, expected_test = unbiased(canonical, **FAKE_PARAMS)

    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


def test_unknown_strategy_raises_listing_the_valid_ones(canonical):
    with pytest.raises(
        ValueError, match="temporal.*unbiased|unbiased.*temporal"
    ):
        train_test_split(canonical, strategy="stratified")


# --- parameter resolution from the loader -----------------------------------


def test_unbiased_reads_parameters_from_the_loader(loader, canonical):
    """The loader is the only carrier of the published per-dataset constants."""
    train, test = train_test_split(loader, strategy="unbiased")
    expected_train, expected_test = unbiased(canonical, **FAKE_PARAMS)

    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


def test_explicit_parameter_overrides_the_loader(loader, canonical):
    train, test = train_test_split(loader, strategy="unbiased", max_days=2.0)
    expected_train, expected_test = unbiased(
        canonical, start_date=None, end_date=None, max_days=2.0
    )

    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


def test_unbiased_on_a_plain_dataframe_without_params_raises(canonical):
    """No loader means no published constants — say so, never guess."""
    with pytest.raises(ValueError, match="max_days"):
        train_test_split(canonical, strategy="unbiased")


def test_unbiased_on_a_loader_without_params_raises(flat):
    loader = _FakeLoaderNoParams(flat)

    with pytest.raises(ValueError, match="max_days"):
        train_test_split(loader, strategy="unbiased")


def test_unbiased_only_parameters_rejected_by_temporal_strategy(canonical):
    """Silently ignoring max_days would hide a misunderstanding."""
    with pytest.raises(ValueError, match="max_days"):
        train_test_split(canonical, strategy="temporal", max_days=5.0)


# --- degenerate inputs fail loudly ------------------------------------------


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
@pytest.mark.parametrize("strategy", ["temporal", "unbiased"])
def test_test_size_outside_the_unit_interval_raises(canonical, strategy, bad):
    """``test_size=1.5`` used to index past the end and return a bogus split."""
    kwargs = {} if strategy == "temporal" else {"max_days": 5.0}

    # Matched precisely: the pre-existing "empty side" guard also mentions
    # test_size, and would otherwise mask a missing range check.
    with pytest.raises(ValueError, match="test_size must be"):
        train_test_split(canonical, strategy=strategy, test_size=bad, **kwargs)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
def test_primitives_validate_test_size_too(canonical, bad):
    """The check belongs to the primitives, so every caller inherits it."""
    with pytest.raises(ValueError, match="test_size must be"):
        temporal(canonical, test_size=bad)
    with pytest.raises(ValueError, match="test_size must be"):
        unbiased(canonical, None, None, 5.0, test_size=bad)


def test_unbiased_filtering_every_case_raises_a_named_error(canonical):
    """A max_days that keeps nothing must say so, not IndexError."""
    with pytest.raises(ValueError, match="no cases|empty"):
        train_test_split(canonical, strategy="unbiased", max_days=0.0001)


# --- bring-your-own-log ergonomics ------------------------------------------


def test_column_mapping_promotes_a_log_with_custom_column_names(flat):
    """The documented "your own CSV, no to_event_log call" path.

    Without a passthrough the user hits an error telling them to pass
    ``column_mapping``, which the signature would not accept.
    """
    renamed = flat.rename(
        columns={
            elc.case_id: "CaseID",
            elc.timestamp: "Timestamp",
            elc.activity: "Activity",
        }
    )

    train, test = train_test_split(
        renamed,
        column_mapping={
            "case_id": "CaseID",
            "timestamp": "Timestamp",
            "activity": "Activity",
        },
    )

    expected_train, expected_test = train_test_split(to_event_log(flat))
    assert train.index.equals(expected_train.index)
    assert test.index.equals(expected_test.index)


def test_column_mapping_with_a_loader_raises(loader):
    """An EventLog normalized its columns already — a mapping would be a no-op."""
    with pytest.raises(ValueError, match="column_mapping cannot be applied"):
        train_test_split(loader, column_mapping={"case_id": "whatever"})


# --- documented semantics of the temporal strategy --------------------------


def test_temporal_test_size_is_a_fraction_of_the_time_span_not_of_cases():
    """Pins what ``test_size`` actually means for ``strategy="temporal"``.

    The cutoff is placed a fraction of the way through the log's calendar
    span, so with uneven case arrivals the resulting case counts are not
    ``test_size`` of the total. ``unbiased`` is the case-proportional one.
    """
    rows = []
    for case in range(100):
        # Arrivals accelerate, so calendar-fraction != case-fraction.
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(case**1.6 / 50)
        )
        for step in range(3):
            rows.append(
                {
                    elc.case_id: f"s{case:03d}",
                    elc.timestamp: start + pd.Timedelta(hours=step),
                    elc.activity: "a",
                }
            )
    log = to_event_log(pd.DataFrame(rows))

    _, test = train_test_split(log, strategy="temporal", test_size=0.2)

    test_fraction = test.index.get_level_values("case_id").nunique() / 100
    assert test_fraction < 0.2  # far fewer than a naive reading suggests


# --- LogLike parity ---------------------------------------------------------


@pytest.mark.parametrize("strategy", ["temporal", "unbiased"])
def test_all_three_log_forms_split_identically(flat, strategy):
    """EventLog, flat DataFrame and canonical DataFrame agree."""
    kwargs = {} if strategy == "temporal" else dict(FAKE_PARAMS)
    inputs = [
        EventLog(dataframe=flat),
        flat,
        to_event_log(flat),
    ]

    results = [
        train_test_split(log, strategy=strategy, **kwargs) for log in inputs
    ]

    first_train, first_test = results[0]
    for train, test in results[1:]:
        assert train.index.equals(first_train.index)
        assert test.index.equals(first_test.index)


# --- the loader's parameter dict is not shared mutable state -----------------


def test_loader_params_cannot_be_mutated_through_the_property(loader):
    """The property must hand back a copy, not the shared class dict."""
    loader.unbiased_split_params["max_days"] = 999.0

    assert loader.unbiased_split_params["max_days"] == FAKE_PARAMS["max_days"]
    assert _FakeLoader._unbiased_split_params["max_days"] == (
        FAKE_PARAMS["max_days"]
    )


def test_loader_without_params_raises_a_named_error(flat):
    loader = _FakeLoaderNoParams(flat)

    with pytest.raises(ValueError, match="Unbiased split not available"):
        _ = loader.unbiased_split_params


# --- the split-first workflow is the one that survives cross-validation -----


def test_split_output_works_with_sklearn_cross_validation(canonical):
    """Why splitting first is the documented headline.

    ``train`` is a DataFrame, so sklearn can index it — meta-estimators such
    as ``GridSearchCV`` call ``indexable`` / ``_safe_indexing`` on X before
    any skpm code runs, and an ``EventLog`` satisfies neither (it has no
    ``iloc``, ``__len__`` or ``shape``). Splitting first means every
    downstream step receives a frame.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    from skpm import case_ids
    from skpm.feature_extraction import TimestampExtractor
    from skpm.feature_extraction.targets import remaining_time
    from skpm.sequence_encoding import Aggregation

    train, _ = train_test_split(canonical)
    y_train = remaining_time(train, time_unit="h")

    pipe = Pipeline(
        [
            (
                "time",
                TimestampExtractor(case_features=None, event_features="all"),
            ),
            ("encode", Aggregation(method="mean")),
            ("model", HistGradientBoostingRegressor(random_state=0)),
        ]
    )

    scores = cross_val_score(
        pipe,
        train,
        y_train,
        cv=GroupKFold(n_splits=3),
        groups=case_ids(train),
        scoring="neg_mean_absolute_error",
    )
    assert len(scores) == 3
    assert np.isfinite(scores).all()


# --- the target stays a separate, per-side decision -------------------------


def test_target_is_unchanged_by_splitting_after_instead_of_before(canonical):
    """Both splits are case-level and every target is case-local, so the
    split never needs to bundle y to stay leakage-free."""
    from skpm.feature_extraction.targets import (
        execution_time,
        next_activity,
        remaining_time,
    )

    train, test = train_test_split(canonical)

    for target in (remaining_time, next_activity, execution_time):
        after = pd.concat([target(train), target(test)])
        before = target(canonical).loc[after.index]
        assert before.equals(after)
