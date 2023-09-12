import pytest
import numpy as np
import pandas as pd
import datetime as dt
from skpm.encoding import TraceAggregator

def test_trace_encoding():
    dummy_data = pd.DataFrame(
        {
            "case_id": np.repeat(np.arange(0, 10), 100),
            "activity": np.random.randint(0, 10, 1000),     #.astype(np.object_),
            "resource": np.random.randint(0, 3, 1000)       #.astype(np.object_)
        }
    )
    
    rp = TraceAggregator().set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]
    
    rp = TraceAggregator(method="sum").set_output(transform="pandas")
    rp.fit(dummy_data)
    out = rp.transform(dummy_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == dummy_data.shape[0]
    
    with pytest.raises(Exception) as exc_info: 
        rp.transform(dummy_data[["activity", "resource"]])
    # these asserts are identical; you can use either one   
    assert exc_info.value.args[0].endswith("input dataframe.")