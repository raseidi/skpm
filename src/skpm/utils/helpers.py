from functools import wraps
import pandas as pd
import polars as pl


# def flatten_list(l: list):
#     return [item for sublist in l for item in sublist]

def auto_convert_dataframes(func):
    """
    A decorator that automatically converts DataFrame objects to Pandas DataFrame if necessary.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    """
    if len(func.__qualname__.split('.')) > 1:  # check if decorator is used on a method function
        def wrapper(self, df, *args, **kwargs):
            """
            Method wrapper function that checks if the input DataFrame is a Polars DataFrame.
            If it is, it converts it to a pandas DataFrame, executes the function,
            and converts the result back to a Polars DataFrame if necessary.

            Args:
                self: class instance
                df (DataFrame): The input DataFrame.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                DataFrame: The result DataFrame, possibly converted to Polars format.

            """
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
                result = func(self, df, *args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    result = pl.from_pandas(result)
                return result  # returns polars df
            result = func(self, df, *args, **kwargs)
            return result
    else:

        def wrapper(df, *args, **kwargs):
            """
            Wrapper function that checks if the input DataFrame is a Polars DataFrame.
            If it is, it converts it to a pandas DataFrame, executes the function,
            and converts the result back to a Polars DataFrame if necessary.

            Args:
                df (DataFrame): The input DataFrame.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                DataFrame: The result DataFrame, possibly converted to Polars format.

            """
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
                result = func(df, *args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    result = pl.from_pandas(result)
                return result  # returns polars df
            result = func(df, *args, **kwargs)
            return result

    return wrapper


@auto_convert_dataframes
def infer_column_types(df, int_as_cat=False) -> tuple:
    """Infer column types from a dataframe."""
    cat_cols = ["object", "category"]
    if int_as_cat:
        cat_cols.append("int")
    cat = df.select_dtypes(include=cat_cols).columns.tolist()
    num = df.select_dtypes(include=["number"]).columns.tolist()
    time = df.select_dtypes(
        include=[
            "datetime",
            "datetime64",
            "datetimetz",
            "datetime64[ns]",
            "timedelta",
            "timedelta64",
        ]
    ).columns.tolist()

    # remove the int columns from num if int_as_cat is True
    if int_as_cat:
        num = list(set(num) - set(cat))

    return cat, num, time
