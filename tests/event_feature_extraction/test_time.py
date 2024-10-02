import numpy as np
import pandas as pd
import datetime as dt

import pytest
from skpm.event_feature_extraction import TimestampExtractor
from skpm.config import EventLogConfig as elc

def test_time():
    dummy_data = pd.DataFrame(
        {
            elc.case_id: [1, 1, 1, 2, 2, 2],
            elc.timestamp: [
                dt.datetime(2021, 1, 1, 0, 0, 0),
                dt.datetime(2021, 1, 1, 0, 0, 1),
                dt.datetime(2021, 1, 1, 0, 0, 2),
                dt.datetime(2021, 1, 1, 0, 12, 0),
                dt.datetime(2021, 1, 1, 0, 12, 1),
                dt.datetime(2021, 1, 1, 0, 12, 3),
            ],
        }
    )

    # test TimeStampExtractor
    t = TimestampExtractor()
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == t._n_features_out
    assert isinstance(out, pd.DataFrame)

    dummy_data = pd.DataFrame(
        {
            elc.case_id: [1, 1, 1, 2, 2, 2],
            elc.timestamp: ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", ""],
        }
    )
    t = TimestampExtractor()
    with pytest.raises(Exception) as exc_info:
        t.fit(dummy_data[[elc.case_id, elc.timestamp]])
