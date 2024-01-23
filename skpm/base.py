from pandas import DataFrame
from sklearn.base import BaseEstimator

from .config import EventLogConfig as elc
from .utils.validation import ensure_list, validate_columns


class BaseProcessEstimator(BaseEstimator):
    """Base class for all process estimators in skpm.

    This class implements a common interface for all process,
    aiming at standardizing the validation and transformation
    of event logs.

    For instance, all event logs must have a `case_id` column.
    """

    def _validate_log(self, X: DataFrame, copy: bool = True):
        # TODO: the validation of a dataframe might be done
        # through the `pd.api.extensions`.
        # This would decrease the dependency between data validation
        # and sklearn estimators.
        # See: https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-pandas
        data = X.copy() if copy else X

        # TODO: only when we perform groupby on np.array
        # X = super()._validate_data(X)

        # despite the bottlenecks, event logs are better handled as dataframes
        assert isinstance(data, DataFrame), "Input must be a dataframe."
        cols = ensure_list(data.columns)

        if not self._ensure_case_id(data.columns):
            raise ValueError(f"Column `{elc.case_id}` not found.")

        if cols:
            cols = validate_columns(
                input_columns=data.columns,
                required=[elc.case_id] + self.features_,
            )
        else:
            cols = data.columns

        return data[cols]

    def _ensure_case_id(self, columns: list[str]):
        for col in columns:
            if col.endswith(elc.case_id):
                elc.case_id = col
                return True
        return False
