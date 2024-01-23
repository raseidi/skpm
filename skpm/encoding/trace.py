from functools import partial
from sklearn.base import (
    TransformerMixin,
    check_is_fitted,
)
from sklearn.utils._param_validation import StrOptions
from skpm.base import BaseProcessEstimator

from skpm.config import EventLogConfig as elc
from skpm.utils.helpers import infer_column_types

class Aggregation(TransformerMixin, BaseProcessEstimator):
    """Sequence Encoding Transformer.

    This module implements a method for encoding sequences by
    aggregating features. It adapts the approach from a
    research paper [1] that abstracts event sequences by
    disregarding their order and using aggregation functions.
    Common aggregation functions include frequency-based
    methods for categorical features and general statistics
    (average, sum, etc.) for numeric attributes.

    In our implementation, we assume that categorical
    features are already encoded categorical features and
    apply aggregation methods accordingly: frequency
    aggregation for integer (categorical) features and
    general statistical measures for float (numerical)
    features. This design choice allows flexibility in
    aggregating user-engineered features, not limited to
    one-hot encoding as described in the original
    paper [1].


    Parameters
    ----------
    num_method : str, default="mean"
        The method to aggregate numerical features.
        Possible values: "sum", "mean".
    cat_method : str, default="sum"
        The method to aggregate categorical features.
        Possible values: "frequency", "sum".

    Attributes
    ----------
    n_features_ : int
        The number of features to encode.
    features_ : list[str]
        The features to encode.
    cat_ : list[str]
        The categorical features to encode.
    num_ : list[str]
        The numerical features to encode.

    References
    ----------
    [1] Outcome-Oriented Predictive Process Monitoring: Review and Benchmark, Teinemaa, I., Dumas, M., Maggi, F. M., & La Rosa, M. (2019).

    Examples
    --------
    >>> import numpy as npd
    >>> import pandas as pd
    >>> from skpm.encoding import Aggregation
    >>> df = pd.DataFrame({
    ...     "timestamp": np.arange(10),
    ...     "activity": np.random.randint(0, 10, 10),
    ...     "resource": np.random.randint(0, 3, 10),
    ...     "case_id": np.random.randint(0, 3, 10)
    ... }).sort_values(by=["case_id", "timestamp"])
    >>> df = pd.get_dummies(df, columns=[elc.activity, elc.resource], dtype=int)
    >>> df = df.drop("timestamp", axis=1)
    >>> Aggregation().fit_transform(df)
    """

    _parameter_constraints = {
        "num_method": [
            StrOptions({"sum", "mean", "median"}),
        ],
        "cat_method": [
            StrOptions({"sum", "mean", "median", }),
        ],
    }

    def __init__(
        self,
        num_method="mean",
        cat_method="sum",
        # n_jobs=1,
    ) -> None:
        self.num_method = num_method
        self.cat_method = cat_method
        # self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self._validate_params()
        self.features_ = X.columns.drop(elc.case_id).tolist()
        self.n_features_ = len(self.features_)

        X = self._validate_log(X)

        # TODO: we assume int as categorical (e.g. one-hot)
        # if cat_cols:
        #     self.cat_ = cat_cols
        # if num_cols:
        #     self.num_ = num_cols
        self.cat_, self.num_, _ = infer_column_types(X[self.features_], int_as_cat=True)
        
        self.feature_aggregations_ = {
            **{cat_col: self.cat_method for cat_col in self.cat_},
            **{num_col: self.num_method for num_col in self.num_},
        }
        
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_")
        X = self._validate_log(X)

        group = X.groupby(elc.case_id, observed=True, as_index=False)
    
        # TODO: major bottleneck; polars is 50x faster for this operation       
        X[self.features_] = (
            group
            .expanding()
            .agg(
                self.feature_aggregations_
            )
            .values
        )
        return X

class WindowAggregation(Aggregation):
    def __init__(self, window_size=2, min_events=1, num_method="mean", cat_method="sum") -> None:
        self.num_method = num_method
        self.cat_method = cat_method
        self.window_size = window_size
        self.min_events = min_events        
        
    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_")
        X = self._validate_log(X)

        group = (
            X
            .groupby(
                elc.case_id, 
                observed=True, 
                as_index=False)
            .rolling(
                window=self.window_size, 
                min_periods=self.min_events)
        )
        
        X[self.features_] = (
            group
            .agg(self.feature_aggregations_)
            .values
        )
        return X