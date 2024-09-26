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


@pytest.mark.timeout(300)
def test_bpi():
    # TODO: deactivate warnings from pm4py
    # TODO: the download is fast but the reading (pm4py) is slow

    # testing BPI13ClosedProblems (smallest log, faster to test)
    default_path = "data/BPI13ClosedProblems/BPI_Challenge_2013_closed_problems.parquet"
    if os.path.exists(default_path):
        os.remove(default_path)
    bpi = BPI13ClosedProblems()

    assert isinstance(bpi.log, pd.DataFrame)
    assert isinstance(bpi.__repr__(), str)
    assert isinstance(len(bpi.log), int)

    # covering pytest when the file already exists
    bpi = BPI13ClosedProblems(bpi.file_path)

    # bpi19 covers a diff case of tests
    bpi19 = BPI19()
    assert isinstance(bpi19.log, pd.DataFrame)

    # removing downloaded files
    import shutil

    shutil.rmtree("data/BPI13ClosedProblems")
    if os.path.dirname(bpi19.file_path) != os.getcwd():
        shutil.rmtree(os.path.dirname(bpi19.file_path))

# from skpm.event_logs import *
# from skpm.event_logs import split

# datasets = [
#     BPI12,
#     BPI13ClosedProblems,
#     BPI13Incidents,
#     BPI13OpenProblems,
#     BPI17,
#     BPI19,
#     BPI20PrepaidTravelCosts,
#     BPI20TravelPermitData,
#     BPI20RequestForPayment,
#     BPI20DomesticDeclarations,
#     BPI20InternationalDeclarations,
#     Sepsis,
# ]

# for d in datasets:
#     print(d.__name__)
#     log = d()
#     print(log.log.shape)
#     try:
#         train, test = split.unbiased(log.log, **log.unbiased_split_params)
#         print(train.shape, test.shape)
#     except Exception as e:
#         print(e)
        
#     print("=="*20)