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

    Structurally-missing cells are padded with ``fill_value`` (``0`` by
    default → a NaN-free matrix); pass ``None`` to keep ``NaN``. See
    :class:`_PositionalEncoder`.
    """

    _parameter_constraints = {
        **_PositionalEncoder._parameter_constraints,
        "n": [Interval(type=Integral, left=1, right=None, closed="left")],
    }

    _position_label = "w"

    def __init__(self, n: int = 2, fill_value: float | int | str | None = 0):
        super().__init__(fill_value=fill_value)
        self.n = n

    def _positions(self, X: pd.DataFrame) -> list[int]:
        return list(range(self.n))

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "attributes_")

        group = X.groupby(level=self.case_id, sort=False, observed=True)

        # Build each attribute's window block, then assemble once to avoid
        # fragmenting the frame with per-column inserts.
        frames = []
        for col in self.attributes_:
            window_cols = [
                f"{col}_{self._position_label}_{pos}" for pos in self.positions_
            ]
            shifted = group[col].shift(
                self.positions_, fill_value=self.fill_value
            )
            shifted.columns = window_cols
            frames.append(shifted)

        if not frames:
            return pd.DataFrame(index=X.index)
        return pd.concat(frames, axis=1)
