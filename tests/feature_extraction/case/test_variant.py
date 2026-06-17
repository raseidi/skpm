import numpy as np
import pandas as pd

import pytest
from skpm.feature_extraction.case import VariantExtractor
from skpm.config import EventLogConfig as elc


def test_variants():
    n_cases = 100
    np.random.seed(42)
    dummy_data = pd.DataFrame(
        {
            elc.case_id: np.random.randint(0, n_cases, 1000),
            elc.activity: np.random.randint(0, 10, 1000),
            elc.timestamp: pd.date_range("2024-01-01", periods=1000, freq="h"),
        }
    )

    rp = VariantExtractor()
    rp.fit(dummy_data)
    df = rp.transform(dummy_data)
    assert df.variant.nunique() == n_cases

    inv_t = rp.inverse_transform(df.variant)
    assert inv_t.shape == (n_cases,)
    assert isinstance(inv_t[0], tuple)


def test_variants_equal_length_cases():
    # Regression: when all cases share a length, the per-case activity tuples
    # form a rectangular block that LabelEncoder/np.unique coerces into a 2-D
    # array and crashes on (unhashable ndarray). pd.factorize keeps the
    # variant tuples hashable.
    df = pd.DataFrame(
        {
            elc.case_id: np.repeat([0, 1, 2, 3], 3),
            elc.activity: [
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
                "x",
                "c",
                "a",
                "b",
                "c",
            ],
            elc.timestamp: pd.date_range("2024-01-01", periods=12, freq="h"),
        }
    )

    rp = VariantExtractor()
    out = rp.fit_transform(df)
    assert len(out) == 4
    assert out.variant.nunique() == 2  # (a, b, c) and (a, x, c)

    inv = rp.inverse_transform(out.variant)
    assert inv.shape == (4,)
    assert isinstance(inv[0], tuple)
    assert ("a", "x", "c") in list(inv)
    assert ("a", "b", "c") in list(inv)
