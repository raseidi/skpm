from numbers import Integral

import pandas as pd
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from skpm.sequence_encoding._positional import _PositionalEncoder


class Windowing(_PositionalEncoder):
    """Sliding-window encoding per case — a fixed-width view of recent events.

    For each event, build columns ``<attr>_w_0, <attr>_w_1, ..., <attr>_w_{n-1}``
    holding the value of ``attr`` ``j`` steps back within the case
    (``groupby(level="case_id").shift(j)``). ``w_0`` is the current event,
    ``w_1`` the previous one, etc., so the columns are an order-preserving
    window over the most recent ``n`` events (the current one included). The
    first ``j`` events of each case have no value at ``w_j`` and are padded.

    Unlike :class:`Aggregation` (which collapses the window into summary
    statistics), the window keeps each event in its own columns, preserving
    order. Use :class:`Indexing` instead for absolute, position-from-start
    encoding.

    Padding follows the shared contract (see :class:`_PositionalEncoder`):
    numeric columns are zero-padded, categoricals use ``fill_cat_value`` (``0``
    by default), datetimes use ``NaT``; pass ``None`` to keep ``NaN``.
    """

    _parameter_constraints = {
        **_PositionalEncoder._parameter_constraints,
        "n": [Interval(type=Integral, left=1, right=None, closed="left")],
    }

    _position_label = "w"

    def __init__(
        self,
        n: int = 2,
        attributes: str | list[str] | None = None,
        fill_cat_value: int | str | None = 0,
        fill_num_value: float | None = 0.0,
    ):
        super().__init__(
            attributes=attributes,
            fill_cat_value=fill_cat_value,
            fill_num_value=fill_num_value,
        )
        self.n = n

    def _positions(self, X: pd.DataFrame) -> list[int]:
        return list(range(self.n))

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "attributes_")

        group = X.groupby(level=self.case_id, sort=False, observed=True)
        num_attributes = X.select_dtypes(include=float).columns
        time_attributes = X.select_dtypes(
            include=["datetime", "timedelta", "datetimetz"]
        ).columns

        # Build each attribute's window block, then assemble once to avoid
        # fragmenting the frame with per-column inserts.
        frames = []
        for col in self.attributes_:
            fill_value = self._fill_value(col, num_attributes, time_attributes)
            window_cols = [
                f"{col}_{self._position_label}_{pos}" for pos in self.positions_
            ]
            shifted = group[col].shift(self.positions_, fill_value=fill_value)
            shifted.columns = window_cols
            frames.append(shifted)

        if not frames:
            return pd.DataFrame(index=X.index)
        return pd.concat(frames, axis=1)
