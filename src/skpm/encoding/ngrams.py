from numpy import ndarray
import pandas as pd
from typing import Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer
import warnings


def _trace_to_ngram(trace: Union[list, np.array], N: int = 3) -> list:
    if not isinstance(trace, np.ndarray):
        trace = np.array(trace, dtype=np.int32)

    trace = np.insert(trace, 0, -1)  # TODO: special token here
    trace = np.append(trace, -1)  # otherwise, it only works for integers
    gram_list = []
    for ix in range(len(trace) - N + 1):
        gram = tuple(trace[ix : ix + N])
        gram_list.append(gram)
    return gram_list


def traces_to_ngrams(
    traces: list[Union[list, np.array]], N: int = 3
) -> tuple[list, set]:
    """
    Returns a list of n-grams and a set of unique n-grams.

    Parameters
    ----------
    traces : list[list[object]]
    N : int, optional. N-gram size. Default: 3

    Returns
    -------
    traces_as_ngrams : list of list of n-grams
        A list of traces, where each trace is a list of n-grams.

    unique_grams : set of n-grams
        A set of unique n-grams.
    """
    traces_as_ngrams = [_trace_to_ngram(t, N) for t in traces]
    # flatten and set to get unique n-grams
    unique_grams = set([gram for trace in traces_as_ngrams for gram in trace])
    return traces_as_ngrams, unique_grams


class EncodedNgrams(TransformerMixin, BaseEstimator):
    def __init__(self, N: int = 3) -> None:
        super().__init__()
        self.N = N

    def get_feature_names_out(self):
        return [f"{self.N}-grams"]

    def fit(self, X, y=None):
        ngrams = X.groupby("caseid").activity.apply(_trace_to_ngram, N=self.N)
        # cant use _unique from sklearn.utils._encode
        # because it only works for 1D arrays
        unique_ngrams = set([gram for trace in ngrams for gram in trace])
        self.vocab_ngrams_ = {gram: ix for ix, gram in enumerate(unique_ngrams)}
        # self.ngrams = ngrams.explode().reset_index()

        # self.ngrams = ngrams.activity.map(self.vocab_ngrams)
        return self

    def transform(self, X, y=None):
        if not self._check_is_fitted():
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        ngrams = X.groupby("caseid").activity.apply(_trace_to_ngram, N=self.N)
        unique_ngrams = set([gram for trace in ngrams for gram in trace])

        # check for new ngrams
        self.new_ngrams_ = unique_ngrams - set(self.vocab_ngrams_.keys())
        if self.new_ngrams_:
            warnings.warn(
                f"Found {len(self.new_ngrams_)} new n-grams. Call `self.new_ngrams_` to see them."
            )

        # args to control this behavior
        ngrams = ngrams.explode().reset_index()
        ngrams.activity = ngrams.activity.map(self.vocab_ngrams_)

        return ngrams

    def _check_is_fitted(self):
        return hasattr(self, "vocab_ngrams_")

    # def get_feature_names_out(self, input_features=None):


dummy_log = pd.DataFrame(
    {
        "caseid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "activity": [10, 20, 30, 10, 20, 30, 10, 20, 30],
    }
)

ng = EncodedNgrams(N=2).fit(dummy_log)
ng.transform(dummy_log)
