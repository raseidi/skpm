import pandas as pd
from skpm.config import EventLogConfig as elc


def _bounded_dataset(
    dataset: pd.DataFrame, start_date, end_date: int
) -> pd.DataFrame:
    grouped = dataset.groupby(elc.case_id, as_index=False)[elc.timestamp].agg(
        ["min", "max"]
    )

    start_date = (
        pd.Period(start_date)
        if start_date
        else dataset[elc.timestamp].min().to_period("M")
    )
    end_date = (
        pd.Period(end_date)
        if end_date
        else dataset[elc.timestamp].max().to_period("M")
    )
    bounded_cases = grouped[
        (grouped["min"].dt.to_period("M") >= start_date)
        & (grouped["max"].dt.to_period("M") <= end_date)
    ][elc.case_id].values
    dataset = dataset[dataset[elc.case_id].isin(bounded_cases)]
    return dataset


def _unbiased(dataset: pd.DataFrame, max_days: int) -> pd.DataFrame:
    grouped = (
        dataset.groupby(elc.case_id, as_index=False)[elc.timestamp]
        .agg(["min", "max"])
        .assign(
            duration=lambda x: (x["max"] - x["min"]).dt.total_seconds()
            / (24 * 60 * 60)
        )
    )

    # condition 1: cases are shorter than max_duration
    condition_1 = grouped["duration"] <= max_days * 1.00000000001
    # condition 2: drop cases starting after the dataset's last timestamp - the max_duration
    latest_start = dataset[elc.timestamp].max() - pd.Timedelta(
        max_days, unit="D"
    )
    condition_2 = grouped["min"] <= latest_start

    unbiased_cases = grouped[condition_1 & condition_2][elc.case_id].values
    dataset = dataset[dataset[elc.case_id].isin(unbiased_cases)]
    return dataset


def unbiased(
    dataset: pd.DataFrame,
    start_date: str | pd.Period | None,
    end_date: str | pd.Period | None,
    max_days: int,
    test_len: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unbiased split of event log into training and test set [1].

    Code adapted from [2].

    Parameters
    ----------
    dataset: pd.DataFrame
        Event log.

    start_date: str
        Start date of the event log.

    end_date: str
        End date of the event log.

    max_days: int
        Maximum duration of cases.

    test_len: float, default=0.2
        Proportion of cases to be used for the test set.

    Returns
    -------
    - df_train: pd.DataFrame, training set
    - df_test: pd.DataFrame, test set

    Example
    -------
    >>> from skpm.event_logs import BPI12
    >>> from skpm.event_logs import split
    >>> bpi12 = BPI12()
    >>> df_train, df_test = split.unbiased(bpi12.log, **bpi12.unbiased_split_params)
    >>> df_train.shape, df_test.shape
    ((117546, 7), (55952, 7))

    References:
    ----------
    [1] Hans Weytjens, Jochen De Weerdt. Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring, 2021. doi: 10.1007/978-3-030-94343-1_2
    [2] https://github.com/hansweytjens/predictive-process-monitoring-benchmarks
    """
    dataset[elc.timestamp] = pd.to_datetime(
        dataset[elc.timestamp], utc=True
    ).dt.tz_localize(None)

    # bounding the event log
    if start_date or end_date:
        dataset = _bounded_dataset(dataset, start_date, end_date)

    # drop longest cases and debiasing end of dataset
    dataset = _unbiased(dataset, max_days)

    # preliminaries
    grouped = dataset.groupby(elc.case_id, as_index=False)[elc.timestamp].agg(
        ["min", "max"]
    )

    ### TEST SET ###
    first_test_case_nr = int(len(grouped) * (1 - test_len))
    first_test_start_time = (
        grouped["min"].sort_values().values[first_test_case_nr]
    )
    # retain cases that end after first_test_start time
    test_case_nrs = grouped.loc[
        grouped["max"].values >= first_test_start_time, elc.case_id
    ]
    df_test = dataset[dataset[elc.case_id].isin(test_case_nrs)].reset_index(
        drop=True
    )

    #### TRAINING SET ###
    df_train = dataset[~dataset[elc.case_id].isin(test_case_nrs)].reset_index(
        drop=True
    )

    return df_train, df_test
