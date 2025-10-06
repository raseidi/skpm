import polars as pl
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from skpm.config import EventLogConfigMixin

from .utils.validation import ensure_list, validate_columns

__all__ = ["BaseProcessEstimator", "BaseProcessTransformer"]

class BaseProcessEstimator(BaseEstimator, EventLogConfigMixin):
    """Base class for all process estimators in SkPM.

    This class implements a common interface for all process,
    aiming at standardizing the validation and transformation
    of event logs.

    """
    _requires_case_id: bool = True
    def _validate_log(
        self,
        X: DataFrame | pl.DataFrame,
        y: DataFrame | pl.DataFrame | None = None,
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
        self._validate_params()
        
        is_polars = False
        if isinstance(X, pl.DataFrame):  # For Polars DataFrame
            X = X.to_pandas()
            is_polars = True

        # TODO: the validation of a dataframe might be done
        # through the `pd.api.extensions`.
        # This would decrease the dependency between data validation
        # and sklearn estimators.
        # See: https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-pandas
        data = X.copy() if copy else X

        # despite the bottlenecks, event logs are better handled as dataframes
        assert isinstance(data, DataFrame), "Input must be a dataframe."
        cols = ensure_list(data.columns)

        if self._requires_case_id:
            self._case_id = self._ensure_case_id(data.columns)

        # cols = validate_columns(
        #     input_columns=data.columns,
        #     required=[self._case_id] + list(self.feature_names_in_),
        # )

        if is_polars:  # For Polars DataFrame
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
            if col.endswith(self.case_id):
                return col
        raise ValueError(f"Case ID column not found.")
    
class BaseProcessTransformer(TransformerMixin, BaseProcessEstimator):
    def fit(self, X, y=None):
        self._validate_log(X)
        
        self._fit(X, y)
        return self
    
    def transform(self, X, y=None):
        out = self._transform(X, y)
        
        return out
                
    def _fit(self, X, y=None):
        return self
    
    def _transform(self, X, y=None):
        raise NotImplementedError("Abstract Base Method")