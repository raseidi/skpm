import polars as pl
import pytest
import numpy as np
import pandas as pd
from skpm.sequence_encoding import Indexing
from skpm.config import EventLogConfig as elc


@pytest.fixture(name="pd_df")
def fixture_dummy_pd():
    return pd.DataFrame(
        {
            elc.case_id: np.repeat(np.arange(0, 10), 100),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.resource: np.random.randint(0, 3, 1000),
        }
    )

def test_indexing(pd_df):
    # Test default Indexing
    rp = Indexing(n=2, attributes=[elc.activity, elc.resource], fill_value=0)
    rp.fit(pd_df)
    out = rp.transform(pd_df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == pd_df.shape[0]

    with pytest.raises(Exception):
        rp.transform(pd_df[[elc.activity, elc.resource]])

    rp = Indexing(n=2, attributes=elc.activity, fill_value=0)
    rp.fit(pd_df)
    
    rp = Indexing(n=2, attributes=None, fill_value=0)
    rp.fit(pd_df)