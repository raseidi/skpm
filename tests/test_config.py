import pandas as pd
import pytest

from skpm.config import EventLogConfig, EventLogConfigMixin, _EventLogConfig


@pytest.fixture(autouse=True)
def _reset_config():
    EventLogConfig.reset_global_config()
    yield
    EventLogConfig.reset_global_config()


def test_singleton_identity_via_mixin():
    class A(EventLogConfigMixin):
        pass

    a, b = A(), A()
    assert a._config is b._config is EventLogConfig
    assert isinstance(EventLogConfig, _EventLogConfig)


def test_set_and_reset_global_config():
    defaults = EventLogConfig.get_global_config()
    assert defaults["case_id"] == "case:concept:name"

    EventLogConfig.set_global_config(case_id="CaseID", activity="Activity")
    assert EventLogConfig.case_id == "CaseID"
    assert EventLogConfig.activity == "Activity"
    # Untouched keys preserve their defaults.
    assert EventLogConfig.timestamp == defaults["timestamp"]

    # Mixin reads through the singleton.
    class A(EventLogConfigMixin):
        pass

    assert A().case_id == "CaseID"

    EventLogConfig.reset_global_config()
    assert EventLogConfig.get_global_config() == defaults


def test_normalize_columns_noop_when_standards_present():
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
    assert list(out.columns) == list(df.columns)


def test_normalize_columns_renames_via_mapping():
    df = pd.DataFrame(
        {
            "CaseID": [1, 1, 2],
            "Activity": ["a", "b", "a"],
            "Timestamp": ["2024-01-01"] * 3,
            "Resource": ["r1", "r2", "r1"],
            "extra": [0, 0, 0],
        }
    )
    mapping = {
        "case_id": "CaseID",
        "activity": "Activity",
        "timestamp": "Timestamp",
        "resource": "Resource",
    }
    out = EventLogConfig.normalize_columns(df, mapping=mapping)
    assert EventLogConfig.case_id in out.columns
    assert EventLogConfig.activity in out.columns
    assert EventLogConfig.timestamp in out.columns
    assert EventLogConfig.resource in out.columns
    assert "extra" in out.columns
    # Input is not mutated.
    assert "CaseID" in df.columns


def test_normalize_columns_partial_mapping_keeps_existing_standards():
    df = pd.DataFrame(
        {
            "case:concept:name": [1],
            "Activity": ["a"],
            "time:timestamp": ["2024-01-01"],
        }
    )
    out = EventLogConfig.normalize_columns(df, mapping={"activity": "Activity"})
    assert EventLogConfig.activity in out.columns
    assert "Activity" not in out.columns


def test_normalize_columns_resource_optional():
    df = pd.DataFrame(
        {
            "case:concept:name": [1],
            "concept:name": ["a"],
            "time:timestamp": ["2024-01-01"],
        }
    )
    out = EventLogConfig.normalize_columns(df)
    assert EventLogConfig.resource not in out.columns


def test_normalize_columns_raises_on_missing_required():
    df = pd.DataFrame({"concept:name": ["a"], "time:timestamp": ["2024-01-01"]})
    with pytest.raises(ValueError, match="case_id"):
        EventLogConfig.normalize_columns(df)


def test_normalize_columns_raises_when_mapping_points_to_missing_column():
    df = pd.DataFrame({"concept:name": ["a"], "time:timestamp": ["2024-01-01"]})
    with pytest.raises(ValueError, match="case_id"):
        EventLogConfig.normalize_columns(df, mapping={"case_id": "nope"})


def _make_log(case_col, activity_col, timestamp_col):
    return pd.DataFrame(
        {
            case_col: [1, 1, 2, 2, 3],
            activity_col: ["a", "b", "a", "c", "a"],
            timestamp_col: pd.to_datetime(
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


from skpm.base import BaseProcessTransformer


class _DummyTransformer(BaseProcessTransformer):
    """Minimal transformer used by snapshot tests."""

    _parameter_constraints: dict = {}

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        return pd.DataFrame({"resolved_case_id": [self.case_id] * len(X)})


def _build_dummy_transformer():
    return _DummyTransformer()


def test_snapshot_taken_at_fit():
    df = _make_log("case:concept:name", "concept:name", "time:timestamp")
    t = _build_dummy_transformer()
    assert not hasattr(t, "case_id_")
    t.fit(df)
    assert t.case_id_ == "case:concept:name"
    assert t.activity_ == "concept:name"
    assert t.timestamp_ == "time:timestamp"
    assert t.resource_ == "org:resource"


def test_global_change_after_fit_does_not_affect_fitted_estimator():
    df = _make_log("case:concept:name", "concept:name", "time:timestamp")
    t = _build_dummy_transformer().fit(df)
    EventLogConfig.set_global_config(case_id="totally_different")
    assert t.case_id == "case:concept:name"
    out = t.transform(df)
    assert out["resolved_case_id"].iloc[0] == "case:concept:name"


def test_property_falls_back_to_global_before_fit():
    t = _build_dummy_transformer()
    EventLogConfig.set_global_config(case_id="CaseID")
    assert t.case_id == "CaseID"
    EventLogConfig.set_global_config(case_id="other")
    assert t.case_id == "other"


def test_pipeline_each_step_snapshots_independently():
    from sklearn.pipeline import Pipeline

    df = _make_log("case:concept:name", "concept:name", "time:timestamp")

    step_a = _build_dummy_transformer()
    EventLogConfig.set_global_config(case_id="first_case")
    step_a.fit(df.rename(columns={"case:concept:name": "first_case"}))
    snapshot_a = step_a.case_id_

    step_b = _build_dummy_transformer()
    EventLogConfig.set_global_config(case_id="second_case")
    step_b.fit(df.rename(columns={"case:concept:name": "second_case"}))
    snapshot_b = step_b.case_id_

    assert snapshot_a == "first_case"
    assert snapshot_b == "second_case"
    # Step A is unaffected by the global change that happened before B fit.
    assert step_a.case_id == "first_case"

    # And a real Pipeline can chain them.
    EventLogConfig.reset_global_config()
    pipe = Pipeline([("a", _build_dummy_transformer())])
    pipe.fit(_make_log("case:concept:name", "concept:name", "time:timestamp"))
    assert pipe.named_steps["a"].case_id_ == "case:concept:name"


def test_clone_drops_snapshot():
    from sklearn.base import clone

    df = _make_log("case:concept:name", "concept:name", "time:timestamp")
    fitted = _build_dummy_transformer().fit(df)
    assert hasattr(fitted, "case_id_")

    fresh = clone(fitted)
    assert not hasattr(fresh, "case_id_")
    # Pre-fit, the clone reads the live global default.
    assert fresh.case_id == EventLogConfig.case_id


def test_pickled_estimator_keeps_its_snapshot():
    import pickle

    df = _make_log("case:concept:name", "concept:name", "time:timestamp")
    fitted = _build_dummy_transformer().fit(df)
    blob = pickle.dumps(fitted)

    EventLogConfig.set_global_config(case_id="changed_after_pickle")
    restored = pickle.loads(blob)
    assert restored.case_id == "case:concept:name"


def test_event_log_normalizes_on_load():
    from skpm.event_logs.base import EventLog

    df = pd.DataFrame(
        {
            "CaseID": [1, 1, 2],
            "Activity": ["a", "b", "a"],
            "Timestamp": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
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
    # case_id and timestamp moved into the event-log MultiIndex.
    assert tuple(log.dataframe.index.names) == ("case_id", "timestamp", "event_id")
    assert EventLogConfig.activity in log.dataframe.columns
