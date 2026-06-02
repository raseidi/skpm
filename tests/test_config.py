import pandas as pd
import pytest

from skpm.base import BaseProcessTransformer
from skpm.config import EventLogConfig, EventLogConfigMixin, _EventLogConfig


@pytest.fixture(autouse=True)
def _reset_config():
    EventLogConfig.reset_global_config()
    yield
    EventLogConfig.reset_global_config()


# --- singleton + fixed canonical names --------------------------------------


def test_singleton_identity_via_mixin():
    class A(EventLogConfigMixin):
        pass

    a, b = A(), A()
    assert a._config is b._config is EventLogConfig
    assert isinstance(EventLogConfig, _EventLogConfig)


def test_canonical_names_are_fixed_constants():
    assert EventLogConfig.case_id == "case_id"
    assert EventLogConfig.timestamp == "timestamp"
    assert EventLogConfig.activity == "activity"
    assert EventLogConfig.resource == "resource"
    # Registering an alias does NOT change the canonical output name.
    EventLogConfig.set_global_config(activity="Activity")
    assert EventLogConfig.activity == "activity"


# --- alias registry ---------------------------------------------------------


def test_set_and_reset_source_names():
    defaults = EventLogConfig.get_global_config()
    assert defaults["case_id"] == "case:concept:name"  # XES default

    EventLogConfig.set_global_config(case_id="CaseID", activity="Activity")
    cfg = EventLogConfig.get_global_config()
    assert cfg["case_id"] == "CaseID"
    assert cfg["activity"] == "Activity"
    # Configuring source names does not change the canonical output names.
    assert EventLogConfig.case_id == "case_id"

    EventLogConfig.reset_global_config()
    assert EventLogConfig.get_global_config() == defaults


# --- normalize_columns: resolution ------------------------------------------


def test_normalize_columns_canonicalizes_xes_names():
    df = pd.DataFrame(
        {
            "case:concept:name": [1, 1, 2],
            "concept:name": ["a", "b", "a"],
            "time:timestamp": ["2024-01-01"] * 3,
            "org:resource": ["r1", "r2", "r1"],
            "extra": [0, 0, 0],
        }
    )
    out = EventLogConfig.normalize_columns(df)
    assert set(out.columns) == {"case_id", "timestamp", "activity", "resource", "extra"}
    assert "extra" in out.columns  # untouched columns preserved
    assert "case:concept:name" in df.columns  # input not mutated


def test_normalize_columns_noop_when_canonical_present():
    df = pd.DataFrame(
        {
            "case_id": [1, 2],
            "timestamp": ["2024-01-01", "2024-01-02"],
            "activity": ["a", "b"],
        }
    )
    out = EventLogConfig.normalize_columns(df)
    assert list(out.columns) == list(df.columns)


def test_normalize_columns_via_global_config():
    # The user declares their source naming once; columns are then recognized.
    df = pd.DataFrame(
        {"CaseID": [1], "Activity": ["a"], "Timestamp": ["2024-01-01"]}
    )
    EventLogConfig.set_global_config(
        case_id="CaseID", activity="Activity", timestamp="Timestamp"
    )
    out = EventLogConfig.normalize_columns(df)
    assert {"case_id", "timestamp", "activity"} <= set(out.columns)


def test_normalize_columns_does_not_guess_nonstandard_names():
    # Without declaring (set_global_config) or mapping, non-XES / non-canonical
    # names are NOT guessed — the user must declare them.
    df = pd.DataFrame(
        {"CaseID": [1], "Activity": ["a"], "Timestamp": ["2024-01-01"]}
    )
    with pytest.raises(ValueError, match="case_id"):
        EventLogConfig.normalize_columns(df)


def test_normalize_columns_via_explicit_mapping():
    df = pd.DataFrame(
        {"cid": [1], "act": ["a"], "ts": ["2024-01-01"], "extra": [0]}
    )
    out = EventLogConfig.normalize_columns(
        df, mapping={"case_id": "cid", "activity": "act", "timestamp": "ts"}
    )
    assert set(out.columns) == {"case_id", "timestamp", "activity", "extra"}
    assert "cid" in df.columns  # input not mutated


def test_normalize_columns_resource_optional():
    df = pd.DataFrame(
        {
            "case:concept:name": [1],
            "concept:name": ["a"],
            "time:timestamp": ["2024-01-01"],
        }
    )
    out = EventLogConfig.normalize_columns(df)
    assert "resource" not in out.columns


def test_normalize_columns_raises_on_missing_required():
    df = pd.DataFrame({"concept:name": ["a"], "time:timestamp": ["2024-01-01"]})
    with pytest.raises(ValueError, match="case_id"):
        EventLogConfig.normalize_columns(df)


def test_normalize_columns_raises_on_invalid_mapping():
    df = pd.DataFrame(
        {
            "case:concept:name": [1],
            "concept:name": ["a"],
            "time:timestamp": ["2024-01-01"],
        }
    )
    with pytest.raises(ValueError, match="nope"):
        EventLogConfig.normalize_columns(df, mapping={"case_id": "nope"})


# --- fit-time behavior ------------------------------------------------------


class _DummyTransformer(BaseProcessTransformer):
    _parameter_constraints: dict = {}

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        return pd.DataFrame({"resolved_case_id": [self.case_id] * len(X)})


def _make_log():
    return pd.DataFrame(
        {
            "case:concept:name": [1, 1, 2, 2, 3],
            "concept:name": ["a", "b", "a", "c", "a"],
            "time:timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
        }
    )


def test_estimator_uses_canonical_names():
    from skpm.event_logs.base import to_event_log

    canonical = to_event_log(_make_log())  # promote with default (XES) config
    t = _DummyTransformer().fit(canonical)
    # Canonical names are fixed constants — no fit-time snapshot.
    assert t.case_id == "case_id"
    assert not hasattr(t, "case_id_")
    # Changing the source-name config afterwards does not affect the canonical
    # names, nor a fitted estimator operating on already-canonical data.
    EventLogConfig.set_global_config(case_id="something_else")
    assert t.case_id == "case_id"
    out = t.transform(canonical)
    assert out["resolved_case_id"].iloc[0] == "case_id"


def test_event_log_normalizes_on_load():
    from skpm.event_logs.base import EventLog

    df = pd.DataFrame(
        {
            "CaseID": [1, 1, 2],
            "Activity": ["a", "b", "a"],
            "Timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    log = EventLog(
        dataframe=df,
        column_mapping={
            "case_id": "CaseID",
            "activity": "Activity",
            "timestamp": "Timestamp",
        },
    )
    # case_id and timestamp moved into the event-log MultiIndex; activity stays
    # as a canonical-named column.
    assert tuple(log.dataframe.index.names) == ("case_id", "timestamp", "event_id")
    assert "activity" in log.dataframe.columns
