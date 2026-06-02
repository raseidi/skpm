import pandas as pd
import polars as pl
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from skpm.config import EventLogConfigMixin
from skpm.event_logs.base import EVENT_LOG_INDEX, has_event_log_index, to_event_log

__all__ = ["BaseProcessEstimator", "BaseProcessTransformer"]


class BaseProcessEstimator(BaseEstimator, EventLogConfigMixin):
    """Base class for all process estimators in SkPM.

    Estimators expect the input event log to carry the canonical
    ``MultiIndex(case_id, timestamp, event_id)`` (see
    :func:`skpm.event_logs.base.to_event_log`). If a caller hands in a
    flat DataFrame, :meth:`_validate_log` promotes it on the fly, so
    ad-hoc usage and pipelines that ingest unprocessed frames still work.

    Polars input is accepted as a legacy path: it is *not* promoted to a
    MultiIndex (polars has no equivalent shape) and is forwarded as-is
    to engine-specific transform implementations.
    """

    def _validate_log(
        self,
        X: DataFrame | pl.DataFrame,
        y: DataFrame | pl.DataFrame | None = None,
        copy: bool = True,
    ) -> DataFrame | pl.DataFrame:
        """Validate, normalize, and return ``X`` in event-log form."""
        self._validate_params()

        if isinstance(X, pl.DataFrame):
            raise NotImplementedError(
                "Polars DataFrames are not yet supported. Please convert to pandas DataFrame."
            ) # ToDo: removing polars support for now

        if not isinstance(X, DataFrame):
            raise TypeError(
                f"Input must be a pandas or polars DataFrame, got {type(X).__name__}."
            )

        if copy:
            X = X.copy()

        if not has_event_log_index(X):
            X = to_event_log(X)

        return X

    @staticmethod
    def _case_ids(X: DataFrame) -> pd.Series:
        """Return the ``case_id`` index level as a Series aligned to ``X.index``."""
        return X.index.get_level_values("case_id").to_series(
            index=X.index, name="case_id"
        )

    @staticmethod
    def _timestamps(X: DataFrame) -> pd.Series:
        """Return the ``timestamp`` index level as a Series aligned to ``X.index``."""
        return X.index.get_level_values("timestamp").to_series(
            index=X.index, name="timestamp"
        )

    @staticmethod
    def _event_ids(X: DataFrame) -> pd.Series:
        """Return the ``event_id`` index level as a Series aligned to ``X.index``."""
        return X.index.get_level_values("event_id").to_series(
            index=X.index, name="event_id"
        )


class BaseProcessTransformer(TransformerMixin, BaseProcessEstimator):
    def fit(self, X, y=None):
        self._snapshot_config()
        self._validate_log(X)

        self._fit(X, y)
        return self

    def transform(self, X, y=None):
        out = self._transform(X, y)
        return out

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        raise NotImplementedError("Abstract Base Method")
