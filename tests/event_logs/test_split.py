import os
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from skpm.event_logs import BPI20PrepaidTravelCosts
from skpm.event_logs.split import unbiased, temporal

def test_temporal():
    with TemporaryDirectory() as tmpdirname:
        bpi = BPI20PrepaidTravelCosts(cache_folder=tmpdirname)

        train, test = temporal(bpi)
        
        assert train.shape[0] > test.shape[0] # training set larger than test set
        assert train.shape[1] == bpi.dataframe.shape[1] # same number of columns
        assert train.shape[0] + test.shape[0] == bpi.dataframe.shape[0] # same number of events
        
        with pytest.raises(ValueError):
            bpi = BPI20PrepaidTravelCosts(bpi.file_path[0])


def test_unbiased():
    with TemporaryDirectory() as tmpdirname:
        bpi = BPI20PrepaidTravelCosts(cache_folder=tmpdirname)

        train, test = unbiased(bpi, **bpi.unbiased_split_params)
        
        assert train.shape[0] > test.shape[0] # training set larger than test set
        assert train.shape[1] == bpi.dataframe.shape[1] # same number of columns
        assert train.shape[0] + test.shape[0] < bpi.dataframe.shape[0] # unbiased split reduces the number of events
        
        with pytest.raises(ValueError):
            bpi = BPI20PrepaidTravelCosts(bpi.file_path[0])
