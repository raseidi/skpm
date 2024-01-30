import os
import pytest
import numpy as np
from skpm.event_logs.download import download_url
from skpm.event_logs.extract import extract_gz
from skpm.utils import validation as v


def test_validation():
    with pytest.raises(Exception):
        v.validate_columns(input_columns=[1, 2, 3], required=[4])

    out = v.ensure_list("exception")
    assert isinstance(out, list)

    out = v.ensure_list({1, 2, 3})
    assert isinstance(out, list)
