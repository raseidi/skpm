import numpy as np
import pandas as pd
from skpm.event_feature_extraction import WorkInProgress
from skpm.config import EventLogConfig as elc


def test_wip():
    dummy_log = pd.DataFrame(
        {
            elc.case_id: np.random.randint(1, 10, 100),
            elc.timestamp: pd.date_range("2021-01-01", periods=100, freq="6h"),
            elc.activity: np.random.choice(["a", "b", "c"], 100),
        }
    ).sort_values(elc.timestamp)
    wip = WorkInProgress()
    wip.fit(dummy_log)
    wip_values = wip.transform(dummy_log)
    assert isinstance(wip_values, np.ndarray)
    assert wip_values.shape == (len(dummy_log),)

    wip = (
        WorkInProgress()
        .set_output(transform="pandas")
        .fit(dummy_log)
        .transform(dummy_log)
    )
    assert isinstance(wip, pd.DataFrame)
