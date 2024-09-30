import polars as pl
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
        """
        Validate and preprocess the input event log DataFrame.

        Parameters
        ----------
        X : DataFrame
            The input DataFrame representing the event log.
        y : DataFrame, default=None
            The target DataFrame associated with the event log.
        reset : bool, default=True
            Whether to reset the index of the DataFrame after validation.
        cast_to_ndarray : bool, default=False
            Whether to cast the DataFrame to a NumPy ndarray after validation.
        copy : bool, default=True
            Whether to create a copy of the DataFrame before validation.

        Returns
        -------
        DataFrame
            The preprocessed and validated event log DataFrame.

        Raises
        ------
        ValueError
            If the input is not a DataFrame or if the case ID column is missing.
        """
        polar_df = False
        if isinstance(X, pl.DataFrame):  # For Polars DataFrame
            X = X.to_pandas()
            polar_df = True

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

        case_id = self._ensure_case_id(data.columns)

        self._validate_data(
            X=X.drop(columns=case_id, axis=1),
            y=y,
            reset=reset,
            cast_to_ndarray=cast_to_ndarray,
        )

        # if cols:
        cols = validate_columns(
            input_columns=data.columns,
            required=[case_id] + self.features_,
        )
        # else:
        #     cols = data.columns

        if polar_df:  # For Polars DataFrame
            data = pl.from_pandas(data)
        return data[cols]

    def _ensure_case_id(self, columns: list[str]):
        """
        Ensure that the case ID column is present in the list of columns.

        Parameters
        ----------
        columns : list[str]
            The list of column names to check for the presence of the case ID.

        Returns
        -------
        bool
            True if the case ID column is found, False otherwise.
        """
        for col in columns:
            if col.endswith(elc.case_id):
                return col
        raise ValueError(f"Case ID column not found.")
