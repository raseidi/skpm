import numpy as np
import pandas as pd
import datetime as dt
from skpm import event_feature_extraction


def test():
    dummy_data = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 2, 2, 2],
            "timestamp": [
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
    t = event_feature_extraction.Timestamp(
        case_col="case_id", time_col="timestamp", features="execution_time"
    )
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == 1
    assert isinstance(out, pd.DataFrame)