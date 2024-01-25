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

    def _validate_log(
        self,
        X: DataFrame,
        y: DataFrame = None,
        reset: bool = True,
        cast_to_ndarray: bool = False,
        copy: bool = True,
    ):
        self._validate_params()

        # TODO: the validation of a dataframe might be done
        # through the `pd.api.extensions`.
        # This would decrease the dependency between data validation
        # and sklearn estimators.
        # See: https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-pandas
        data = X.copy() if copy else X

        # despite the bottlenecks, event logs are better handled as dataframes
        assert isinstance(data, DataFrame), "Input must be a dataframe."
        cols = ensure_list(data.columns)

        if not self._ensure_case_id(data.columns):
            raise ValueError(f"Column `{elc.case_id}` not found.")

        self._validate_data(
            X=X.drop(columns=elc.case_id, axis=1),
            y=y,
            reset=reset,
            cast_to_ndarray=cast_to_ndarray,
        )

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
