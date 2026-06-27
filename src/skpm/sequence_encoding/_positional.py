from numbers import Real

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class _PositionalEncoder(BaseProcessTransformer):
    """Shared machinery for per-case positional encoders.

    Resolves which attributes to encode, picks a per-column padding value,
    and builds the output column names ``<attr>_<label>_<position>``.
    Subclasses declare their position set (:meth:`_positions`) and how each
    cell is filled (:meth:`_transform`).

    Structurally-missing cells (positions outside a case) are padded with
    ``fill_num_value`` / ``fill_cat_value`` rather than left as ``NaN``, so
    the output is directly usable by estimators that reject NaN (e.g.
    ``GradientBoostingRegressor``). The defaults zero-pad numeric columns
    (``0.0``) and use ``0`` as the categorical padding token; datetime columns
    are always padded with ``NaT``. Pass ``None`` for a fill to keep ``NaN``
    (e.g. to handle missingness downstream with an imputer).
    """

    _parameter_constraints = {
        "attributes": [str, list, None],
        "fill_cat_value": [int, str, None],
        "fill_num_value": [Real, None],
    }

    #: Column-name infix distinguishing each encoder's output (``pos``/``w``).
    _position_label = "pos"

    def __init__(
        self,
        attributes: str | list[str] | None = None,
        fill_cat_value: int | str | None = 0,
        fill_num_value: float | None = 0.0,
    ):
        self.attributes = attributes
        self.fill_cat_value = fill_cat_value
        self.fill_num_value = fill_num_value

    def _fit(self, X: pd.DataFrame, y=None):
        # Resolve attributes without touching the param, so the estimator
        # survives clone/get_params and a second transform is stable.
        if self.attributes is None:
            self.attributes_ = X.columns.tolist()
        elif isinstance(self.attributes, str):
            self.attributes_ = [self.attributes]
        else:
            self.attributes_ = list(self.attributes)

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

    def _fill_value(self, col, num_attributes, time_attributes):
        if col in num_attributes:
            return self.fill_num_value
        if col in time_attributes:
            return None
        return self.fill_cat_value

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)
