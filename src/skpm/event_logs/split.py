import pandas as pd
import numpy as np
from skpm.event_logs import read_xes


def start_from_date(dataset, start_date):
    """
    removes outliers starting before start date from dataset
    Args:
        dataset: pandas DataFrame
        start_date: string "MM-YYYY": dataset starts here after removing outliers

    Returns:
        dataset: pandas Dataframe

    """
    case_starts_df = pd.DataFrame(
        dataset.groupby("case:concept:name")["time:timestamp"].min().reset_index()
    )
    case_starts_df["date"] = case_starts_df["time:timestamp"].dt.to_period("M")
    cases_after = case_starts_df[case_starts_df["date"].astype("str") >= pd.Period(start_date)][
        "case:concept:name"
    ].values
    dataset = dataset[dataset["case:concept:name"].isin(cases_after)]
    return dataset


def end_before_date(dataset, end_date):
    """

    removes outliers ending after end date from dataset
    Args:
        dataset: pandas DataFrame
        end_date: string "MM-YYYY": dataset stops here after removing outliers

    Returns:
        dataset: pandas Dataframe
    """
    case_stops_df = pd.DataFrame(
        dataset.groupby("case:concept:name")["time:timestamp"].max().reset_index()
    )
    case_stops_df["date"] = case_stops_df["time:timestamp"].dt.to_period("M")
    cases_before = case_stops_df[case_stops_df["date"] <= pd.Period(end_date)][
        "case:concept:name"
    ].values
    dataset = dataset[dataset["case:concept:name"].isin(cases_before)]
    return dataset


def limited_duration(dataset, max_duration):
    """

    limits dataset to cases shorter than maximal duration and debiases the end of the dataset
    by dropping cases starting after the last timestamp of the dataset - max_duration
    Args:
        dataset: pandas DataFrame
        max_duration: float

    Returns:
        dataset: pandas Dataframe
        latest_start: timeStamp with new end time for the dataset

    """
    # compute each case's duration
    agg_dict = {"time:timestamp": ["min", "max"]}
    duration_df = pd.DataFrame(
        dataset.groupby("case:concept:name").agg(agg_dict)
    ).reset_index()
    duration_df["duration"] = (
        duration_df[("time:timestamp", "max")] - duration_df[("time:timestamp", "min")]
    ).dt.total_seconds() / (24 * 60 * 60)
    # condition 1: cases are shorter than max_duration
    condition_1 = duration_df["duration"] <= max_duration * 1.00000000001
    cases_retained = duration_df[condition_1]["case:concept:name"].values
    dataset = dataset[dataset["case:concept:name"].isin(cases_retained)].reset_index(
        drop=True
    )
    # condition 2: drop cases starting after the dataset's last timestamp - the max_duration
    latest_start = dataset["time:timestamp"].max() - pd.Timedelta(
        max_duration, unit="D"
    )
    condition_2 = duration_df[("time:timestamp", "min")] <= latest_start
    cases_retained = duration_df[condition_2]["case:concept:name"].values
    dataset = dataset[dataset["case:concept:name"].isin(cases_retained)].reset_index(
        drop=True
    )
    return dataset, latest_start


def trainTestSplit(dataset, test_len, start_date,end_date, max_days):
    """
    splits the dataset in train and test set, applying strict temporal splitting and
    debiasing the test set
    Args:
        df: pandas DataFrame
        test_len: float: share of cases belonging in test set
        latest_start: timeStamp with new end time for the dataset
    Returns:
        df_train: pandas DataFrame
        df_test: pandas DataFrame
    """
    dataset["time:timestamp"] = pd.to_datetime(dataset["time:timestamp"], utc=True)

    if start_date:
        dataset = start_from_date(dataset, start_date)
    if end_date:
        dataset = end_before_date(dataset, end_date)
    
    dataset.drop_duplicates(inplace=True)
    dataset, latest_start = limited_duration(dataset, max_days)
    
    # preliminaries
    case_starts_df = dataset.groupby("case:concept:name")["time:timestamp"].min()
    case_nr_list_start = case_starts_df.sort_values().index.array
    case_stops_df = dataset.groupby("case:concept:name")["time:timestamp"].max().to_frame()

    ### TEST SET ###
    first_test_case_nr = int(len(case_nr_list_start) * (1 - test_len))
    first_test_start_time = np.sort(case_starts_df.values)[first_test_case_nr]
    # retain cases that end after first_test_start time
    test_case_nrs = case_stops_df[
        case_stops_df["time:timestamp"].values >= first_test_start_time
    ].index.array
    df_test_all = dataset[dataset["case:concept:name"].isin(test_case_nrs)].reset_index(drop=True)

    # drop prefixes in test set that are past latest_start
    df_test = df_test_all[df_test_all["time:timestamp"] <= latest_start]

    #### TRAINING SET ###
    train_case_nrs = case_stops_df[
        case_stops_df["time:timestamp"].values < first_test_start_time
    ].index.array  # added values
    df_train = dataset[dataset["case:concept:name"].isin(train_case_nrs)].reset_index(drop=True)

    return df_train, df_test
    
