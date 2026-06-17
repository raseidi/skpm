import numpy as np
import pandas as pd
import polars as pl
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from skpm.config import EventLogConfigMixin
from skpm.event_logs.base import has_event_log_index, to_event_log

__all__ = [
    "BaseProcessEstimator",
    "BaseProcessTransformer",
    "CaseLevelTransformer",
]


class BaseProcessEstimator(BaseEstimator, EventLogConfigMixin):
    """Base class for all process estimators in SkPM.

    Estimators operate on the canonical event-log shape: a DataFrame whose
    index is the ``MultiIndex(case_id, timestamp, event_id)`` (see
    :func:`skpm.event_logs.base.to_event_log`). :meth:`_validate_log` enforces
    that shape, promoting a flat DataFrame on the fly so ad-hoc callers can
    hand in unprocessed input.

    Polars input is currently rejected (``NotImplementedError``) — temporarily
    unsupported pending a polars path.
    """

    def _validate_log(self, X, copy: bool = True) -> DataFrame:
        """Validate ``X`` and return it in canonical event-log form.

        Promotes a flat DataFrame via :func:`to_event_log`. An
        already-canonical frame is returned as-is (copied when ``copy``).
        """
        self._validate_params()

        if isinstance(X, pl.DataFrame):
            raise NotImplementedError(
                "Polars DataFrames are not yet supported. "
                "Please convert to a pandas DataFrame."
            )
        if not isinstance(X, DataFrame):
            raise TypeError(
                f"Input must be a pandas DataFrame, got {type(X).__name__}."
            )

        if has_event_log_index(X):
            return X.copy() if copy else X
        # to_event_log returns a fresh frame, so no extra copy is needed.
        return to_event_log(X)

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
        """Return the ``event_id`` index level as a Series aligned to ``X.index``.

        ``event_id`` is the globally-unique, per-event row counter assigned
        at load. For an event's position *within its trace*, use
        :meth:`_trace_positions` instead.
        """
        return X.index.get_level_values("event_id").to_series(
            index=X.index, name="event_id"
        )

    @staticmethod
    def _trace_positions(X: DataFrame) -> pd.Series:
        """Return each event's 0-based position within its case (trace position).

        Distinct from ``event_id`` (a globally-unique row counter): this
        restarts at 0 for every case, encoding an event's index within its
        trace (i.e. prefix length minus one). Computed on demand, not stored.
        """
        return (
            X.groupby(level="case_id", sort=False, observed=True)
            .cumcount()
            .rename("trace_position")
        )

    def _record_feature_names_in(self, X: DataFrame) -> None:
        """Record fit-time feature bookkeeping (scikit-learn convention).

        The "features" of an event log are its non-index columns (``activity``,
        ``resource``, case attributes, ...); ``case_id`` / ``timestamp`` live in
        the index. ``n_features_in_`` is always set; ``feature_names_in_`` is set
        only when every column name is a string, matching scikit-learn.
        """
        self.n_features_in_ = X.shape[1]
        columns = list(X.columns)
        if columns and all(isinstance(c, str) for c in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_


class BaseProcessTransformer(TransformerMixin, BaseProcessEstimator):
    """Base class for **event-level** process transformers (row-preserving).

    The output has one row per input event and preserves the event-log
    ``MultiIndex``. Subclasses implement :meth:`_fit` and :meth:`_transform`
    and **must not** override :meth:`fit` / :meth:`transform` — the base
    handles validation (input is already canonical when ``_fit`` / ``_transform``
    receive it), the fitted marker, and the output/feature-name check.

    For transformers that reduce the log to one row per case, subclass
    :class:`CaseLevelTransformer` instead.
    """

    #: "event" (one output row per event) vs "case" (one row per case).
    _cardinality: str = "event"

    def fit(self, X, y=None):
        X = self._validate_log(X)
        self._record_feature_names_in(X)
        self._fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        X = self._validate_log(X, copy=False)
        out = self._transform(X, y)
        self._check_feature_names_out(out)
        return out

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        raise NotImplementedError("Abstract Base Method")

    def _check_feature_names_out(self, out) -> None:
        """Best-effort guard: if ``_transform`` returns a DataFrame and the
        subclass declares ``get_feature_names_out``, the columns must match.

        Silently skipped when ``get_feature_names_out`` is undefined or raises
        (e.g. mixins that need ``feature_names_in_``), so it only ever fires on
        a genuine contributor mistake.
        """
        if not isinstance(out, DataFrame):
            return
        try:
            declared = list(self.get_feature_names_out())
        except Exception:
            return
        if list(out.columns) != declared:
            raise ValueError(
                f"{type(self).__name__}._transform produced columns "
                f"{list(out.columns)} but get_feature_names_out() declares "
                f"{declared}; they must match."
            )


class CaseLevelTransformer(BaseProcessTransformer):
    """Base for **trace-level** transformers that emit one row per case.

    Unlike event-level transformers (which preserve ``n_samples`` and the event
    ``MultiIndex``), a case-level transformer collapses the log to one row per
    case. It is therefore a **terminal** step: it does not compose as an
    intermediate step of a scikit-learn :class:`~sklearn.pipeline.Pipeline`,
    because the change in row count breaks alignment with ``y`` and downstream
    steps. Use it standalone.

    (This replaces the old stack-inspection guard: rather than trying to detect
    Pipeline membership at runtime, the cardinality is declared explicitly and
    misuse fails fast — a following event-level step receives case-indexed data
    and its ``_validate_log`` raises a clear error.)
    """

    _cardinality: str = "case"
