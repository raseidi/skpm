import warnings

import numpy as np
import pandas as pd
import pytest

from skpm.config import EventLogConfig as elc
from skpm.sequence_encoding import Indexing


@pytest.fixture(name="pd_df")
def fixture_dummy_pd():
    n=1000
    start = pd.Timestamp("2000-01-01")
    end = pd.Timestamp("2024-01-01")
    timestamps = pd.to_datetime(
        np.linspace(start.value, end.value, n).astype(np.int64)
    )

    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), int(n/10)),
            elc.activity: np.random.randint(0, 10, n).astype(str),
            elc.resource: np.random.rand(n),
            elc.timestamp: timestamps
        }
    )

def test_indexing(pd_df):
    rp = Indexing()
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]
    
    with pytest.raises(Exception):
        rp = Indexing(n=0)
        rp.fit(pd_df)
        
    rp = Indexing(n=2, fill_cat_value="TEST", fill_num_value=-1)
    rp.fit_transform(pd_df)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        rp = Indexing(n=None)
        rp.fit_transform(pd_df)
    
    rp = Indexing(n=None, attributes="concept:name")
    rp.fit_transform(pd_df)
