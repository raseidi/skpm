from numbers import Integral, Real

import pandas as pd
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class Indexing(BaseProcessTransformer):
    """Position-based encoding per case.

    For each event, build columns ``<attr>_pos_0, <attr>_pos_1, ...,
    <attr>_pos_{n-1}`` holding the value of ``attr`` at a per-case position.
    Both modes emit the same columns; ``mode`` only changes what each
    position means:

    - ``mode="absolute"`` (default): ``pos_j`` is the value of the case's
      ``j``-th event (0-based), revealed only once the prefix has reached it.
      For the event at trace step ``k``, ``pos_j`` is event ``j`` when
      ``j <= k`` and padded otherwise, so a prefix never sees an event beyond
      its own length (no future leakage). This is the classical index-based
      encoding (see Leontjeva et al., 2015, Teinemaa et al., 2019).
    - ``mode="relative"``: ``pos_j`` is the value ``j`` steps back from the
      current event (``groupby(level="case_id").shift(j)``). ``pos_0`` is the
      current event, ``pos_1`` the previous one, etc. — a sliding window over
      the most recent ``n`` events.

    Positions that fall outside the case (before its start in ``relative``,
    or beyond the prefix length in ``absolute``) are structurally missing.
    They are padded with ``fill_num_value`` / ``fill_cat_value`` rather than
    left as ``NaN``, so the output is directly usable by estimators that
    reject NaN (e.g. ``GradientBoostingRegressor``). The defaults zero-pad
    numeric columns (``0.0``) and use ``0`` as the categorical padding token;
    pass other values to customize, or ``None`` to keep ``NaN`` (e.g. to
    handle missingness downstream with an imputer or a NaN-tolerant model).
    Datetime columns are always padded with ``NaT``.
    """

    _parameter_constraints = {
        "n": [Interval(type=Integral, left=1, right=None, closed="left"), None],
        "attributes": [str, list, None],
        "fill_cat_value": [int, str, None],
        "fill_num_value": [Real, None],
        "mode": [StrOptions({"absolute", "relative"})],
    }

    def __init__(
        self,
        n: int | None = 2,
        attributes: str | list[str] | None = None,
        fill_cat_value: int | str | None = 0,
        fill_num_value: float | None = 0.0,
        mode: str = "absolute",
    ):
        self.n = n
        self.attributes = attributes
        self.fill_cat_value = fill_cat_value
        self.fill_num_value = fill_num_value
        self.mode = mode

    def _fit(self, X: pd.DataFrame, y=None):
        # Resolve which attributes to lag without touching the param, so the
        # estimator survives clone/get_params and a second transform is stable.
        if self.attributes is None:
            self.attributes_ = X.columns.tolist()
        elif isinstance(self.attributes, str):
            self.attributes_ = [self.attributes]
        else:
            self.attributes_ = list(self.attributes)

        # Fix the position set at fit so the output feature space is stable.
        # With n=None the count is data-dependent (the longest case), so it
        # must be pinned here rather than recomputed per transform.
        if self.n is not None:
            self.positions_ = list(range(self.n))
        else:
            max_case_len = (
                X.groupby(level=self.case_id, sort=False, observed=True)
                .size()
                .max()
            )
            self.positions_ = list(range(max_case_len))

        self.feature_names_out_ = [
            f"{col}_pos_{pos}"
            for col in self.attributes_
            for pos in self.positions_
        ]
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "attributes_")

        group = X.groupby(level=self.case_id, sort=False, observed=True)
        num_attributes = X.select_dtypes(include=float).columns
        time_attributes = X.select_dtypes(
            include=["datetime", "timedelta", "datetimetz"]
        ).columns

        out = pd.DataFrame(index=X.index)
        if self.mode == "absolute":
            # 0-based position of each event within its case.
            position = group.cumcount()

        for col in self.attributes_:
            if col in num_attributes:
                fill_value = self.fill_num_value
            elif col in time_attributes:
                fill_value = None
            else:
                fill_value = self.fill_cat_value

            if self.mode == "relative":
                pos_cols = [f"{col}_pos_{pos}" for pos in self.positions_]
                shifted = group[col].shift(
                    self.positions_, fill_value=fill_value
                )
                shifted.columns = pos_cols
                for c in pos_cols:
                    out[c] = shifted[c]
            else:  # absolute
                values = X[col]
                for pos in self.positions_:
                    # Freeze the case's pos-th event value and carry it forward
                    # to every later event; events before that position are
                    # padded, so a prefix never sees an event beyond its length.
                    frozen = (
                        values.where(position == pos)
                        .groupby(level=self.case_id, sort=False, observed=True)
                        .ffill()
                    )
                    out[f"{col}_pos_{pos}"] = frozen.where(
                        position >= pos, other=fill_value
                    )

        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)
