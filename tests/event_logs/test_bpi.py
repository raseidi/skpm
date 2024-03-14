import os
import pytest
import pandas as pd
from skpm.event_logs import BPI20, BPI13ClosedProblems, BPI19


@pytest.mark.timeout(300)
def test_bpi():
    # TODO: deactivate warnings from pm4py
    # TODO: the download is fast but the reading (pm4py) is slow

    # not implemented modules
    with pytest.raises(Exception) as exc_info:
        log = BPI20()

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
