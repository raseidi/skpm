import warnings
import numpy as np
import pandas as pd
import datetime as dt

import pytest
from skpm.event_feature_extraction import ResourcePoolExtractor



def test_resource():
    dummy_data = pd.DataFrame(
        {
            "activity": np.random.randint(0, 10, 1000),
            "resource": np.random.randint(0, 3, 1000)
        }
    )

    dummy_data_test = pd.DataFrame(
        {
            "activity": np.random.randint(0, 10, 100),
            "resource": np.random.randint(0, 3, 100)
        }
    )
    
    rp = ResourcePoolExtractor().set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 1
    assert out.columns.tolist() == ["resource_roles"]
    
    rp = ResourcePoolExtractor()
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == dummy_data.shape[0]
    
    test_out = rp.transform(dummy_data_test)
    assert test_out.shape[0] == dummy_data_test.shape[0]
    
    with pytest.raises(Exception) as exc_info:
        dummy_data_test["resource"] = dummy_data_test["resource"].replace(2, np.nan)
        rp.transform(dummy_data_test[["activity", "resource"]])
    
    with pytest.warns():
        dummy_data_test["resource"] = dummy_data_test["resource"].fillna(100)
        test_out = rp.transform(dummy_data_test)
    
    with pytest.raises(Exception) as exc_info:
        dummy_data_test["activity"] = dummy_data_test["activity"].replace(2, np.nan)
        rp.transform(dummy_data_test[["activity", "resource"]])
    