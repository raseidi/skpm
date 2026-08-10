import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from skpm.baselines import ActivityMeanRegressor
from skpm.event_logs.base import to_event_log


@pytest.fixture(name="log")
def fixture_log() -> pd.DataFrame:
    flat = pd.DataFrame(
        {
            "case:concept:name": [1, 1, 1, 2, 2, 3],
            "concept:name": ["a", "b", "c", "a", "b", "a"],
            "time:timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
        }
    )
    return to_event_log(flat)


def test_fit_computes_per_activity_means(log):
    y = np.array(
        [10.0, 20.0, 30.0, 12.0, 22.0, 14.0]
    )  # a:{10,12,14}, b:{20,22}, c:{30}
    reg = ActivityMeanRegressor().fit(log, y)
    check_is_fitted(reg)
    assert reg.activity_means_["a"] == pytest.approx(12.0)
    assert reg.activity_means_["b"] == pytest.approx(21.0)
    assert reg.activity_means_["c"] == pytest.approx(30.0)
    assert reg.global_mean_ == pytest.approx(y.mean())


def test_predict_maps_each_event_to_its_activity_mean(log):
    y = np.array([10.0, 20.0, 30.0, 12.0, 22.0, 14.0])
    reg = ActivityMeanRegressor().fit(log, y)
    preds = reg.predict(log)
    # activities in order a,b,c,a,b,a
    np.testing.assert_allclose(preds, [12.0, 21.0, 30.0, 12.0, 21.0, 12.0])


def test_unseen_activity_falls_back_to_global_mean():
    train = to_event_log(
        pd.DataFrame(
            {
                "case:concept:name": [1, 1, 2],
                "concept:name": ["a", "b", "a"],
                "time:timestamp": pd.date_range(
                    "2024-01-01", periods=3, freq="h"
                ),
            }
        )
    )
    reg = ActivityMeanRegressor().fit(train, np.array([10.0, 20.0, 30.0]))
    test = to_event_log(
        pd.DataFrame(
            {
                "case:concept:name": [9, 9],
                "concept:name": ["a", "zzz"],  # "zzz" unseen at fit
                "time:timestamp": pd.date_range(
                    "2024-02-01", periods=2, freq="h"
                ),
            }
        )
    )
    preds = reg.predict(test)
    assert preds[0] == pytest.approx(20.0)  # mean of 'a' = (10+30)/2
    assert preds[1] == pytest.approx(reg.global_mean_)  # unseen -> global


def test_works_as_pipeline_final_estimator(log):
    # passthrough feature/encoder steps -> the regressor reads the raw log.
    y = np.array([10.0, 20.0, 30.0, 12.0, 22.0, 14.0])
    pipe = Pipeline(
        [
            ("features", "passthrough"),
            ("encoder", "passthrough"),
            ("model", ActivityMeanRegressor()),
        ]
    )
    pipe.fit(log, y)
    assert len(pipe.predict(log)) == len(log)
