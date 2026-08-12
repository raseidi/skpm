from collections.abc import Callable
from numbers import Integral
from typing import Literal, Union

import pandas as pd
import polars as pl
from pandas.api.types import is_numeric_dtype
from sklearn.base import OneToOneFeatureMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


def handle_aggregation_method(method):
    if method == "norm":
        from numpy import linalg

        return linalg.norm
    return method


class Aggregation(OneToOneFeatureMixin, BaseProcessTransformer):
    """Rolling-window aggregation of event features per case.

    Two engines are supported:

    * ``"pandas"`` (default) — operates on an event-log DataFrame
      carrying the canonical ``MultiIndex(case_id, timestamp, event_id)``.
      Grouping uses ``level="case_id"`` and the MultiIndex is preserved
      in the output so downstream steps can reuse it.
    * ``"polars"`` — legacy column-based shape; expects ``case_id`` as a
      regular column (no MultiIndex). Useful when polars throughput
      matters; the output is a flat pandas DataFrame.

    Every column must be numeric — aggregating a string ``activity`` has no
    meaning, so a non-numeric column is an error naming the offending columns
    rather than a silent drop. Encode or select columns the
    sklearn-idiomatic way: one-hot the categoricals in a
    :class:`~sklearn.compose.ColumnTransformer`, or slice the log
    (``log[["duration"]]``) before this step. In a typical pipeline the
    preceding step already emits numbers, e.g.::

        Pipeline([
            ("features", TimestampExtractor(event_features="all")),
            ("encode", Aggregation(method="mean")),
        ])

    Parameters
    ----------
    method : str, default="mean"
        Aggregation method. One of ``"sum"``, ``"mean"``, ``"median"``,
        ``"norm"``.
    prefix_len : int, optional
        Rolling-window length. If None, uses the total number of rows
        (effectively cumulative aggregation per case).
    engine : {"pandas", "polars"}, default="pandas"
        DataFrame backend.

    References
    ----------
    [1] Teinemaa, I., Dumas, M., Maggi, F. M., La Rosa, M. (2019).
        Outcome-Oriented Predictive Process Monitoring: Review and Benchmark.
    """

    _parameter_constraints = {
        "method": [StrOptions({"sum", "mean", "median", "norm"})],
        "prefix_len": [Interval(Integral, 1, None, closed="left"), None],
        "engine": [StrOptions({"pandas", "polars"})],
    }

    def __init__(
        self,
        method: str = "mean",
        prefix_len: int | None = None,
        engine: Literal["pandas", "polars"] = "pandas",
    ) -> None:
        self.method = method
        self.prefix_len = prefix_len
        self.engine = engine

    def _fit(self, X, y=None):
        self._check_numeric(X)
        # prefix_len is the (possibly None) param; prefix_len_ is the window
        # resolved at fit. Keep them distinct so the param survives
        # clone/get_params and check_is_fitted actually detects an unfitted
        # estimator (the param always exists, the fitted attr does not).
        self.prefix_len_ = (
            self.prefix_len if self.prefix_len is not None else len(X)
        )
        return self

    @staticmethod
    def _check_numeric(X) -> None:
        """Reject columns that cannot be aggregated, naming them.

        Without this, a raw event log fails inside pandas with
        ``DataError: Cannot aggregate non-numeric type: object`` — which names
        neither the column nor a way forward, and is not even a ValueError
        subclass, so callers cannot catch it alongside skpm's own errors.

        The predicate is ``is_numeric_dtype``, not ``select_dtypes("number")``:
        the latter is wrong in both directions here, excluding ``bool``
        (which aggregates fine) and including ``timedelta64`` (which does not).
        """
        offenders = {
            str(col): str(dtype)
            for col, dtype in X.dtypes.items()
            if not is_numeric_dtype(dtype)
        }
        if offenders:
            raise ValueError(
                f"Aggregation requires numeric columns, but "
                f"{list(offenders)} cannot be aggregated (dtypes: "
                f"{offenders}). Encode them first (e.g. OneHotEncoder inside "
                f"a ColumnTransformer), or select the numeric columns before "
                f"this step (log[['duration']])."
            )

    def _transform(self, X, y=None):
        check_is_fitted(self, "prefix_len_")
        # X is already a canonical, validated pandas event log (base.transform).
        if self.engine == "polars":
            # polars path kept for future re-enablement (currently unreachable:
            # _validate_log rejects polars input at fit).
            return self._transform_polars(
                pl.from_pandas(X.reset_index())
            ).to_pandas()
        return self._transform_pandas(X)

    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        method_fn = handle_aggregation_method(self.method)
        rolled = (
            X.groupby(level=self.case_id, sort=False, observed=True)
            .rolling(window=self.prefix_len_, min_periods=1)
            .agg(method_fn)
        )
        rolled.index = X.index
        return rolled

    def _transform_polars(self, X: pl.DataFrame) -> pl.DataFrame:
        method_fn = handle_aggregation_method(self.method)
        case_id = self.case_id

        def _make_rolling_expr(col_name: str, fn) -> pl.Expr:
            expr = pl.col(col_name)
            if isinstance(fn, str):
                builtin = f"rolling_{fn}"
                return getattr(expr, builtin)(
                    window_size=self.prefix_len_, min_samples=1
                )
            expr = pl.col(col_name).cast(pl.Float32)
            return expr.rolling_map(
                function=fn, window_size=self.prefix_len_, min_samples=1
            )

        X = X.with_columns(
            [
                _make_rolling_expr(c, method_fn).over(case_id)
                for c in X.columns
                if c != case_id
            ]
        )
        return X.drop(case_id)
