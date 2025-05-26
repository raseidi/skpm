from typing import Literal, Union

import pandas as pd
import polars as pl
from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessEstimator
from skpm.config import EventLogConfig as elc

def handle_aggregation_method(method):
    """Handle the aggregation method.

    Parameters
    ----------
    method : str
        The aggregation method to be handled.

    Returns
    -------
    str, callable
        The aggregation method that pandas or polars can use.
    """
    if method == "norm":
        from numpy import linalg
        return linalg.norm
    return method

class Aggregation(OneToOneFeatureMixin, TransformerMixin, BaseProcessEstimator):
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
    method : str, default="mean"
        The method to aggregate features.
        Possible values: "sum", "mean".
    prefix_len : int, default=None
        The length of the prefix to consider for aggregation. If None, the length of the longest trace is used.
    engine : str, default="pandas"
        The DataFrame engine to use. Supported engines are "pandas" and "polars".

    References
    ----------
    [1] Outcome-Oriented Predictive Process Monitoring: Review and Benchmark, Teinemaa, I., Dumas, M., Maggi, F. M., & La Rosa, M. (2019).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpm.encoding import Aggregation
    >>> from skpm.config import EventLogConfig as elc
    >>> df = pd.DataFrame({
    ...     elc.timestamp: np.arange(10),
    ...     elc.activity: np.random.randint(0, 10, 10),
    ...     elc.resource: np.random.randint(0, 3, 10),
    ...     elc.case_id: np.random.randint(0, 3, 10),
    ... }).sort_values(by=[elc.case_id, elc.timestamp])
    >>> df = pd.get_dummies(df, columns=[elc.activity,elc.resource], dtype=int)
    >>> df = df.drop(elc.timestamp, axis=1)
    >>> Aggregation().fit_transform(df)
    """

    _case_id = elc.case_id
    _parameter_constraints = {
        "method": [
            StrOptions({"sum", "mean", "median", "norm"}),
        ],
        "engine": [
            StrOptions({"pandas", "polars"}),
        ],
    }

    def __init__(
        self,
        method: str = "mean",
        prefix_len: int = None,
        # n_jobs=1,
        engine: Literal[
            "pandas", "polars"
        ] = "pandas",  # Default to Pandas DataFrame
    ) -> None:
        self.method = method
        self.prefix_len = prefix_len
        self.engine = engine

    def validate_engine_with_df(self, X, y=None):
        if (
            self.engine == "pandas"
            and not isinstance(X, pd.DataFrame)
        ):
            X = pd.DataFrame(X)
            y = pd.DataFrame(y) if y is not None else None
        elif (
            self.engine == "polars"
            and not isinstance(X, pl.DataFrame)
        ):
            X = pl.DataFrame(X)
            y = pl.DataFrame(y) if y is not None else None
        return X, y

    def fit(self, X, y=None):
        """Fit transformer.

        Checks if the input is a dataframe, if it
        contains the required columns, validates
        the timestamp column, and the desired features.

        Parameters
        ----------
            X : {DataFrame} of shape (n_samples, n_features+1)
                The data must contain `n_features` plus a column with case ids.
            y : None.
                Ignored.

        Returns
        -------
            self : object
                Fitted aggregator.

        """
        X = self._validate_log(X)

        if self.prefix_len is None:
            self.prefix_len = len(X)

        return self

    def transform(self, X: Union[pd.DataFrame, pl.DataFrame], y=None):
        """Performs the aggregation of event features from a trace.

        Parameters
        ----------
        X : {DataFrame} of shape (n_samples, n_features+1)
            An event log. It must contain n_features + 1 columns,
            representing the case id and the event features.

        Returns
        -------
        X : {DataFrame} of shape (n_samples, n_features)
            The aggregated event log.
        """
        check_is_fitted(self, "n_features_in_")
        X = self._validate_log(X)

        X, y = self.validate_engine_with_df(X, y)
        self._method_fn = handle_aggregation_method(self.method)
        if self.engine == "pandas":  # If using Pandas DataFrame
            if isinstance(X, pl.DataFrame):
                X = X.to_pandas()
            return self._transform_pandas(X)

        else:
            if isinstance(X, pd.DataFrame):
                X = pl.DataFrame(X)
            X = self._transform_polars(X)
            return X.to_pandas()

    def _transform_pandas(self, X: pd.DataFrame):
        """Transforms Pandas DataFrame."""
        group = X.groupby(self._case_id)

        X = (
            group.rolling(window=self.prefix_len, min_periods=1)
            .agg(self._method_fn)
            .reset_index(drop=True)
        )
        return X

    def _transform_polars(self, X: pl.DataFrame):
        """Transforms Polars DataFrame."""
        
        def _make_rolling_expr(col_name: str, method_fn: Union[str, callable]) -> pl.Expr:
            expr = pl.col(col_name)
            
            if isinstance(method_fn, str):
                builtin = f"rolling_{method_fn}"
                fn = getattr(expr, builtin)
                return fn(window_size=self.prefix_len, min_samples=1)
            else:
                expr = pl.col(col_name).cast(pl.Float32)
                return expr.rolling_map(
                    function=method_fn,
                    window_size=self.prefix_len,
                    min_samples=1
                )

        X = X.with_columns([
            _make_rolling_expr(c, self._method_fn).over(self._case_id)
            for c in X.columns
            if c != self._case_id
        ])
 
        return X.drop(self._case_id)
