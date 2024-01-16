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


def test_download():
    url = "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"
    file_name = "BPI_Challenge_2012.xes.gz"
    p = download_url(url, folder="lalala", file_name=file_name)
    exists = os.path.exists(p)
    assert exists

    ep = extract_gz(path=p, folder=os.path.dirname(p))
    extracted_exists = os.path.exists(p.replace(".gz", ""))
    assert extracted_exists

    if exists:
        os.remove(p)

    if extracted_exists:
        os.remove(p.replace(".gz", ""))
