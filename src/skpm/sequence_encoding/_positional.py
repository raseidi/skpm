from numbers import Real

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class _PositionalEncoder(BaseProcessTransformer):
    """Shared machinery for per-case positional encoders.

    Encodes every column of the input into positional columns named
    ``<attr>_<label>_<position>``. Subclasses declare their position set
    (:meth:`_positions`) and how each cell is filled (:meth:`_transform`).
    Select which columns to encode the sklearn-idiomatic way — slice the log
    (``log[["activity"]]``) or wrap the encoder in a ``ColumnTransformer``;
    both preserve the canonical MultiIndex the encoder needs.

    Structurally-missing cells (positions outside a case) are padded with
    ``fill_value`` rather than left as ``NaN``, so the output is directly
    usable by estimators that reject NaN (e.g. ``GradientBoostingRegressor``).
    ``fill_value`` is applied as-is to every column and pandas coerces it per
    dtype (``0`` -> ``0.0`` for floats, an extra category for objects); pass
    ``None`` to keep ``NaN`` (e.g. to impute downstream). Datetime columns are
    rare as encoder inputs (the canonical ``timestamp`` is an index level);
    if present, they too receive ``fill_value``.
    """

    _parameter_constraints = {
        "fill_value": [Real, str, None],
    }

    #: Column-name infix distinguishing each encoder's output (``pos``/``w``).
    _position_label = "pos"

    def __init__(self, fill_value: float | int | str | None = 0):
        self.fill_value = fill_value

    def _fit(self, X: pd.DataFrame, y=None):
        # Encode every input column; column selection is delegated upstream.
        self.attributes_ = X.columns.tolist()

        # Pin the position set at fit so the output feature space is stable.
        self.positions_ = self._positions(X)
        self.feature_names_out_ = [
            f"{col}_{self._position_label}_{pos}"
            for col in self.attributes_
            for pos in self.positions_
        ]
        return self

    def _positions(self, X: pd.DataFrame) -> list[int]:
        """Return the per-case positions to materialise as columns."""
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)
