import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from skpm.config import EventLogConfig as elc


def _trace_to_ngram(trace: Union[list, np.array], N: int = 3) -> list:
    """
    Convert a trace (sequence of events) into n-grams.

    Parameters
    ----------
    trace : list or np.array
        A sequence of events.
    N : int, optional
        Size of the n-grams, by default 3.

    Returns
    -------
    list
        List of n-grams generated from the trace.

    Examples
    --------
    >>> _trace_to_ngram([1, 2, 3, 4, 5], N=2)
    [(-1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, -1)]
    """
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
    Convert a list of traces into n-grams and obtain unique n-grams.

    Parameters
    ----------
    traces : list of list or np.array
        List of traces, where each trace is a sequence of events.
    N : int, optional
        Size of the n-grams, by default 3.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - traces_as_ngrams : a list of list of n-grams
            A list of traces, where each trace is a list of n-grams.
        - unique_grams : set of n-grams
            A set of unique n-grams.
    Examples
    --------
    >>> traces_to_ngrams([[1, 2, 3, 4], [4, 5, 6, 7]], N=2)
    ([[(-1, 1), (1, 2), (2, 3), (3, 4), (4, -1)], [(4, 5), (5, 6), (6, 7), (7, -1)]],
    {(-1, 1), (1, 2), (2, 3), (3, 4), (4, -1), (4, 5), (5, 6), (6, 7), (7, -1)})
    """
    traces_as_ngrams = [_trace_to_ngram(t, N) for t in traces]
    # flatten and set to get unique n-grams
    unique_grams = set([gram for trace in traces_as_ngrams for gram in trace])
    return traces_as_ngrams, unique_grams


class EncodedNgrams(TransformerMixin, BaseEstimator):
    """
    Encode n-grams from a sequence of events.

    Parameters
    ----------
    N : int, optional
        Size of the n-grams, by default 3.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpm.config import EventLogConfig as elc
    >>> dummy_log = pd.DataFrame(
    ...     {
    ...         elc.case_id: [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...         elc.activity: [10, 20, 30, 10, 20, 30, 10, 20, 30],
    ...     }
    ... )
    >>> ng = EncodedNgrams(N=2).fit(dummy_log)
    >>> ng.transform(dummy_log)
       caseid  level_1  activity
    0       1        0         0
    1       1        0         1
    2       1        0         2
    3       2        0         0
    4       2        0         1
    5       2        0         2
    6       3        0         0
    7       3        0         1
    8       3        0         2
    """

    def __init__(self, N: int = 3) -> None:
        super().__init__()
        self.N = N

    def get_feature_names_out(self):
        return [f"{self.N}-grams"]

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self
            Returns the instance itself.
        """
        ngrams = X.groupby(elc.case_id)[elc.activity].apply(_trace_to_ngram, N=self.N)
        # cant use _unique from sklearn.utils._encode
        # because it only works for 1D arrays
        unique_ngrams = set([gram for trace in ngrams for gram in trace])
        self.vocab_ngrams_ = {gram: ix for ix, gram in enumerate(unique_ngrams)}
        # self.ngrams = ngrams.explode().reset_index()

        # self.ngrams = ngrams.activity.map(self.vocab_ngrams)
        return self

    def transform(self, X, y=None):
        """
        Transform the input data into encoded n-grams.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : None
            Ignored.

        Returns
        -------
        DataFrame
             Encoded n-grams.
        """
        if not self._check_is_fitted():
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        ngrams = X.groupby(elc.case_id)[elc.activity].apply(_trace_to_ngram, N=self.N)
        unique_ngrams = set([gram for trace in ngrams for gram in trace])

        # check for new ngrams
        self.new_ngrams_ = unique_ngrams - set(self.vocab_ngrams_.keys())
        if self.new_ngrams_:
            warnings.warn(
                f"Found {len(self.new_ngrams_)} new n-grams. Call `self.new_ngrams_` to see them."
            )

        # args to control this behavior
        ngrams = ngrams.explode().reset_index()
        ngrams[elc.activity] = ngrams[elc.activity].map(self.vocab_ngrams_)

        return ngrams

    def _check_is_fitted(self):
        return hasattr(self, "vocab_ngrams_")

    # def get_feature_names_out(self, input_features=None):


dummy_log = pd.DataFrame(
    {
        elc.case_id: [1, 1, 1, 2, 2, 2, 3, 3, 3],
        elc.activity: [10, 20, 30, 10, 20, 30, 10, 20, 30],
    }
)

ng = EncodedNgrams(N=2).fit(dummy_log)
ng.transform(dummy_log)
