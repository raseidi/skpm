import numpy as np
import pandas as pd

import pytest
from skpm.feature_extraction.case import VariantExtractor
from skpm.config import EventLogConfig as elc


def test_variants():
    n_cases = 100
    dummy_data = pd.DataFrame(
        {
            elc.case_id: np.random.randint(0, n_cases, 1000),
            elc.activity: np.random.randint(0, 10, 1000),
        }
    )

    rp = VariantExtractor()
    rp.fit(dummy_data)
    df = rp.transform(dummy_data)
    assert df.variant.nunique() == n_cases

    inv_t = rp.inverse_transform(df.variant)
    assert inv_t.shape == (n_cases,)
    assert isinstance(inv_t[0], tuple)