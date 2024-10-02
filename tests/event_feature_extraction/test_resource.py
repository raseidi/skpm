import warnings
import numpy as np
import pandas as pd
import datetime as dt

import pytest
from skpm.event_feature_extraction import ResourcePoolExtractor
from skpm.config import EventLogConfig as elc


def test_resource():
    dummy_data = pd.DataFrame(
        {
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )

    dummy_data_test = pd.DataFrame(
        {
            elc.activity: np.random.randint(0, 10, 100),
            elc.resource: np.random.randint(0, 3, 100),
        }
    )

    rp = ResourcePoolExtractor()
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 1
    assert out.columns.tolist() == ["resource_roles"]

    test_out = rp.transform(dummy_data_test)
    assert test_out.shape[0] == dummy_data_test.shape[0]

    with pytest.raises(Exception) as exc_info:
        dummy_data_test[elc.resource] = dummy_data_test[elc.resource].replace(2, np.nan)
        rp.transform(dummy_data_test[[elc.activity, elc.resource]])

    with pytest.warns():
        dummy_data_test[elc.resource] = dummy_data_test[elc.resource].fillna(100)
        test_out = rp.transform(dummy_data_test)

    with pytest.raises(Exception) as exc_info:
        dummy_data_test[elc.activity] = dummy_data_test[elc.activity].replace(2, np.nan)
        rp.transform(dummy_data_test[[elc.activity, elc.resource]])
