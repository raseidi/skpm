import pandas as pd
from sklearn.utils.validation import check_is_fitted

from skpm.sequence_encoding._positional import _PositionalEncoder


class Indexing(_PositionalEncoder):
    """Absolute index-based encoding per case (Leontjeva et al., 2015).

    For each event, build columns ``<attr>_pos_0, <attr>_pos_1, ...`` where
    ``pos_j`` holds the value of ``attr`` at the case's ``j``-th event
    (0-based). The value is revealed only once the prefix has reached that
    position: for the event at trace step ``k``, ``pos_j`` is event ``j`` when
    ``j <= k`` and padded otherwise, so a prefix never sees an event beyond
    its own length (no future leakage).

    The number of positions is the length of the longest case observed at
    fit, so every prefix is fully represented. There is deliberately no ``n``:
    a fixed width shorter than a case would freeze every longer prefix onto
    its first ``n`` events, collapsing them all to a single vector and
    destroying the signal. For a fixed-width sliding window over the most
    recent events, use :class:`Windowing`.

    Padding follows the shared contract (see :class:`_PositionalEncoder`):
    numeric columns are zero-padded, categoricals use ``fill_cat_value`` (``0``
    by default), datetimes use ``NaT``; pass ``None`` to keep ``NaN``.
    """

    _position_label = "pos"

    def _positions(self, X: pd.DataFrame) -> list[int]:
        max_case_len = (
            X.groupby(level=self.case_id, sort=False, observed=True)
            .size()
            .max()
        )
        return list(range(max_case_len))

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "attributes_")

        group = X.groupby(level=self.case_id, sort=False, observed=True)
        num_attributes = X.select_dtypes(include=float).columns
        time_attributes = X.select_dtypes(
            include=["datetime", "timedelta", "datetimetz"]
        ).columns

        # 0-based position of each event within its case.
        position = group.cumcount()

        # Build all columns first and assemble once: with full-width output a
        # per-column insert would fragment the frame (PerformanceWarning).
        columns = {}
        for col in self.attributes_:
            fill_value = self._fill_value(col, num_attributes, time_attributes)
            values = X[col]
            for pos in self.positions_:
                # Freeze the case's pos-th event value and carry it forward to
                # every later event; events before that position are padded, so
                # a prefix never sees an event beyond its own length.
                frozen = (
                    values.where(position == pos)
                    .groupby(level=self.case_id, sort=False, observed=True)
                    .ffill()
                )
                columns[f"{col}_{self._position_label}_{pos}"] = frozen.where(
                    position >= pos, other=fill_value
                )

        return pd.DataFrame(columns, index=X.index)
