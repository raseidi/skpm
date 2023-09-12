from typing import Union

import numpy as np
from pandas import DataFrame
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    check_is_fitted,
)

from skpm.utils import ensure_list, validate_columns


class TraceAggregator(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Sequence encoding transformer.
    Method to encode a sequence of events into a single
    vector. ADAPTED from [1].

    From the paper:
    "Another approach is to consider all events since
    the beginning of the case, but ignore the order of
    the events. This abstraction method paves the way
    to several aggregation functions [...]. Frequency-based
    for activities and general statistics (avg, sum, etc)
    for numeric attributes are the most common aggregation
    functions."

    Parameters
    ----------
    case_col : str, default='case_id'
        Name of the column containing the case ids.
    features : list[str], default=None
        List of features to encode. If None, all features
        are encoded.
    method : str, default='mean'
        The method to encode the sequence of events.
        Supported strategies are "mean" and "sum".
        Usually, the "sum" method is used for
        categorical (frequency) features and the "mean"
        method for numeric features (general statistics).

    Attributes
    ----------
    n_features_ : int
        The number of features to encode.
    features_ : list[str]
        The features to encode.
    method : callable
        The method to encode the sequence of events.

    References
    ----------
    [1] Teinemaa, I., Dumas, M., Maggi, F. M., & La Rosa, M. (2019).

    """

    def __init__(
        self,
        case_col: str = "case_id",
        features: Union[str, list[str]] = None,
        method="mean",
    ) -> None:
        self.case_col = case_col
        self.method = method

        self.features = features

    def fit(self, X, y=None):
        X = self._validate_data(X)

        if not self.features:
            self.features = X.columns.drop(self.case_col).tolist()

        self.n_features_ = len(self.features)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_")
        X = self._validate_data(X)

        # TODO: pandarallel
        group = X.groupby(self.case_col)[self.features]
        if self.method == "mean":
            X = group.expanding().mean()
        elif self.method == "sum":
            X = group.expanding().sum()

        return X

    def _validate_data(self, X):
        self.features = ensure_list(self.features)
        assert isinstance(X, DataFrame), "Input must be a dataframe."
        x = X.copy()

        if not self._ensure_case_id(x.columns):
            raise ValueError(f"Column {self.case_col} is not present in the input dataframe.")

        if self.features:
            cols = validate_columns(
                input_columns=x.columns,
                required=[self.case_col] + self.features,
            )
        else:
            cols = x.columns

        return x[cols]

    def _ensure_case_id(self, columns):
        for col in columns:
            if col.endswith(self.case_col):
                self.case_col = col
                return True
        return False
