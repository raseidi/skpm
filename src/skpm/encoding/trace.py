from typing import Literal, Union

import pandas as pd
import polars as pl
from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessEstimator
from skpm.config import EventLogConfig as elc
from skpm.utils.helpers import infer_column_types

DataFrame = pd.DataFrame
PlDataFrame = pl.DataFrame


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
    num_method : str, default="mean"
        The method to aggregate numerical features.
        Possible values: "sum", "mean".
    cat_method : str, default="sum"
        The method to aggregate categorical features.
        Possible values: "frequency", "sum".
    engine : str, default="pandas"
        The DataFrame engine to use. Supported engines are "pandas" and "polars".

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

    _parameter_constraints = {
        "num_method": [
            StrOptions({"sum", "mean", "median"}),
        ],
        "cat_method": [
            StrOptions(
                {
                    "sum",
                    "mean",
                    "median",
                }
            ),
        ],
    }

    def __init__(
        self,
        num_method: str = "mean",
        cat_method: str = "sum",
        window_size: int = None,
        # n_jobs=1,
        engine: Literal["pandas", "polars"] = "pandas",  # Default to Pandas DataFrame
    ) -> None:
        self.num_method = num_method
        self.cat_method = cat_method
        self.window_size = window_size

        if engine not in ["pandas", "polars"]:
            raise ValueError(
                "Invalid engine. Supported engines are 'pandas' and 'polars'."
            )
        self.engine = engine
        # self.n_jobs = n_jobs

    @staticmethod
    def validate_engine_with_df(func):
        def _decorator(self, *args, **kwargs):
            if (self.engine == "pandas" and not isinstance(args[0], pd.DataFrame)) or (
                self.engine == "polars" and not isinstance(args[0], pl.DataFrame)
            ):
                raise ValueError(
                    "Expected {} dataframe, but received {}".format(
                        self.engine, type(args[0])
                    )
                )
            return func(self, *args, **kwargs)

        return _decorator

    @validate_engine_with_df
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

        cols = list(X.columns)
        cols.remove(elc.case_id)
        self.features_ = cols
        self.n_features_ = len(self.features_)
        X = self._validate_log(X)

        # Infer column types
        self.cat_, self.num_, _ = infer_column_types(X[self.features_], int_as_cat=True)

        self.feature_aggregations_ = {
            cat_col: "sum" if self.cat_method == "sum" else "mean"
            for cat_col in self.cat_
        }
        for num_col in self.num_:
            if self.num_method == "sum":
                self.feature_aggregations_[num_col] = "sum"
            elif self.num_method == "mean":
                self.feature_aggregations_[num_col] = "mean"
            elif self.num_method == "median":
                self.feature_aggregations_[num_col] = "median"

        if self.window_size is None:
            self.window_size = len(X)

        return self

    @validate_engine_with_df
    def transform(self, X: Union[DataFrame, PlDataFrame], y=None):
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
        check_is_fitted(self, "n_features_")
        X = self._validate_log(X, reset=False)

        if self.engine == "pandas":  # If using Pandas DataFrame
            return self._transform_pandas(X)

        elif self.engine == "polars":  # If using Polars DataFrame
            return self._transform_polars(X)

        else:
            raise ValueError(
                "Invalid engine. Supported engines are 'pandas' and 'polars'."
            )

    def _transform_pandas(self, X: DataFrame):
        """Transforms Pandas DataFrame."""
        group = X.groupby(elc.case_id)

        columns = list(self.feature_aggregations_.keys())
        X[columns] = (
            group.rolling(window=self.window_size, min_periods=1)
            .agg(self.feature_aggregations_)
            .values
        )
        return X.drop(elc.case_id, axis=1)

    def _transform_polars(self, X: PlDataFrame):
        """Transforms Polars DataFrame."""
        group = X.group_by(elc.case_id, maintain_order=True)

        for col, method in self.feature_aggregations_.items():
            expanding_expr = getattr(pl.col(col), f"rolling_{method}")(
                window_size=self.window_size, min_periods=1
            )
            expanding_expr = expanding_expr.alias(col)

            out_df = group.agg(expanding_expr)
            out_df = out_df.explode(
                out_df.columns[1:]
            )  # skip case_id; TODO: test when case_id is not the first column
            out_df = out_df.drop(elc.case_id)

            X = X.with_columns(out_df[col].alias(col))
        return X.drop(elc.case_id)
