import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from skpm.encoding.ngrams import _trace_to_ngram, traces_to_ngrams, EncodedNgrams
from skpm.config import EventLogConfig as elc


def test_trace_to_ngram():
    # Test conversion of trace to n-grams
    trace = [1, 2, 3, 4, 5]
    ngrams = _trace_to_ngram(trace, N=2)
    assert ngrams == [(-1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, -1)]


def test_traces_to_ngrams():
    # Test conversion of traces to n-grams
    traces = [[1, 2, 3, 4], [4, 5, 6, 7]]
    ngrams, unique_grams = traces_to_ngrams(traces, N=2)
    expected_ngrams = [
        [(-1, 1), (1, 2), (2, 3), (3, 4), (4, -1)],
        [(-1, 4), (4, 5), (5, 6), (6, 7), (7, -1)],
    ]
    expected_unique_grams = {
        (-1, 1),
        (-1, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, -1),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, -1),
    }
    assert ngrams == expected_ngrams
    assert unique_grams == expected_unique_grams


def test_encoded_ngrams():
    # Test EncodedNgrams transformer
    dummy_log = pd.DataFrame(
        {
            elc.case_id: [1, 1, 1, 2, 2, 2, 3, 3, 3],
            elc.activity: [10, 20, 30, 10, 20, 30, 10, 20, 30],
        }
    )

    # Test fit_transform
    ng = EncodedNgrams(N=2)
    result = ng.fit_transform(dummy_log)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(result), 2)

    # Test transform without fit
    ng_unfitted = EncodedNgrams(N=2)
    with pytest.raises(NotFittedError):
        ng_unfitted.transform(dummy_log)

    # Test new n-grams warning
    dummy_log_extra = pd.DataFrame(
        {
            elc.case_id: [1, 1, 1, 2, 2, 2, 3, 3, 3],
            elc.activity: [
                10,
                20,
                30,
                10,
                20,
                30,
                10,
                20,
                40,
            ],  # Introduce new n-gram (30, 40)
        }
    )
    with pytest.warns(UserWarning):
        ng.transform(dummy_log_extra)
        assert ng.new_ngrams_ == {(20, 40), (40, -1)}
