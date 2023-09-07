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
    features = "execution_time"
    n_features = 1
    t = event_feature_extraction.TimestampExtractor(
        case_col="case_id", time_col="timestamp", features=features
    )
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == n_features
    assert isinstance(out, np.ndarray)

    features = "all"
    t = event_feature_extraction.TimestampExtractor(
        case_col="case_id", time_col="timestamp", features=features
    ).set_output(transform="pandas")
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    # assert out.shape[1] == 1
    assert isinstance(out, pd.DataFrame)

    features = ["execution_time", "remaining_time"]
    n_features = len(features)
    t = event_feature_extraction.TimestampExtractor(
        case_col="case_id", time_col="timestamp", features=features
    ).set_output(transform="pandas")
    t.fit(dummy_data)
    out = t.transform(dummy_data)
    assert out.shape[1] == n_features
    assert isinstance(out, pd.DataFrame)
