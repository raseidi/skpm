import os
import pytest
import pandas as pd
from skpm.event_logs import BPI13ClosedProblems, BPI19

"""Shapes to validate the logs:"""
# BPI12
# (262200, 7)
# (117546, 7) (55952, 7)
# ========================================
# BPI13ClosedProblems
# (6660, 12)
# Unbiased split parameters not supported.
# ========================================
# BPI13Incidents
# (65533, 12)
# Unbiased split parameters not supported.
# ========================================
# BPI13OpenProblems
# (2351, 11)
# Unbiased split parameters not supported.
# ========================================
# BPI17
# (1202267, 19)
# (805809, 19) (294882, 19)
# ========================================
# BPI19
# (1595923, 21)
# (538293, 21) (538880, 21)
# ========================================
# BPI20PrepaidTravelCosts
# (18246, 22)
# (10809, 22) (4928, 22)
# ========================================
# BPI20TravelPermitData
# (86581, 173)
# (51878, 173) (32390, 173)
# ========================================
# BPI20RequestForPayment
# (36796, 14)
# (23295, 14) (7053, 14)
# ========================================
# BPI20DomesticDeclarations
# (56437, 10)
# Unbiased split parameters not supported.
# ========================================
# BPI20InternationalDeclarations
# (72151, 23)
# Unbiased split parameters not supported.
# ========================================
# Sepsis
# (15214, 32)
# Unbiased split parameters not supported.
# ========================================

from tempfile import TemporaryDirectory


def test_bpi():
    with TemporaryDirectory() as tmpdirname:
        bpi = BPI13ClosedProblems(cache_folder=tmpdirname)

        assert isinstance(bpi.dataframe, pd.DataFrame)
        assert isinstance(bpi.__repr__(), str)
        assert isinstance(len(bpi.dataframe), int)

        # covering pytest when the file already exists
        with pytest.raises(ValueError):
            bpi = BPI13ClosedProblems(bpi.file_path[0])
