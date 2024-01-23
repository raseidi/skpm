def flatten_list(l: list):
    return [item for sublist in l for item in sublist]


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
