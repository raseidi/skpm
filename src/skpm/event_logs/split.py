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
    cases_after = case_starts_df[case_starts_df["date"].astype("str") >= start_date][
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
    cases_before = case_stops_df[case_stops_df["date"].astype("str") <= end_date][
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


def trainTestSplit(df, test_len, latest_start, targets):
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
    # preliminaries
    case_starts_df = df.groupby("case:concept:name")["time:timestamp"].min()
    case_nr_list_start = case_starts_df.sort_values().index.array
    case_stops_df = df.groupby("case:concept:name")["time:timestamp"].max().to_frame()

    ### TEST SET ###
    first_test_case_nr = int(len(case_nr_list_start) * (1 - test_len))
    first_test_start_time = np.sort(case_starts_df.values)[first_test_case_nr]
    # retain cases that end after first_test_start time
    test_case_nrs = case_stops_df[
        case_stops_df["time:timestamp"].values >= first_test_start_time
    ].index.array
    df_test_all = df[df["case:concept:name"].isin(test_case_nrs)].reset_index(drop=True)

    # drop prefixes in test set that are past latest_start
    df_test = df_test_all[df_test_all["time:timestamp"] <= latest_start]

    # convert targets into np.NAN for those prefixes that end before the separation time (beginning of test set)
    df_test.loc[df_test["time:timestamp"].values < first_test_start_time, targets] = (
        np.nan
    )

    #### TRAINING SET ###
    train_case_nrs = case_stops_df[
        case_stops_df["time:timestamp"].values < first_test_start_time
    ].index.array  # added values
    df_train = df[df["case:concept:name"].isin(train_case_nrs)].reset_index(drop=True)

    return df_train, df_test


def createClassificationTargets(df, col, keywords_dict):
    # compute target values for classification
    def tv(grp):
        # compute target value (True if one event in group is in keywords)
        for target, keywords in keywords_dict.items():
            grp[target] = True in [x in grp[col].values for x in keywords]
        return grp

    df = df.groupby("case:concept:name").apply(tv)
    return df


def remainTimeOrClassifBenchmark(
    dataset,
    path,
    file_name,
    start_date,
    end_date,
    max_days,
    test_len_share,
    output_type="xes",
    keywords_dict=None,
):
    """
    Creates benchmark training and test sets for remaining time and classification problems

    Args:
        dataset: pandas DataFrame
        path: string: output file path
        file_name: string: output file name
        start_date: string "MM-YYYY": dataset starts here after removing outliers
        end_date: string "MM-YYYY": dataset stops here after removing outliers
        max_days: float
        test_len_share: float: share of cases belonging in test set
        output_type: string: "xes", "pickle" or "csv"
    Returns:
        two files ('_train' and '_test')
    """

    # compute the target
    dataset["time:timestamp"] = pd.to_datetime(dataset["time:timestamp"], utc=True)
    dataset["remain_time"] = (
        dataset.groupby("case:concept:name")["time:timestamp"]
        .apply(lambda x: x.max() - x)
        .values
    )
    dataset["remain_time"] = dataset["remain_time"].dt.total_seconds() / (
        24 * 60 * 60
    )  # convert to float days

    # remove chronological outliers and duplicates
    if start_date:
        dataset = start_from_date(dataset, start_date)
    if end_date:
        dataset = end_before_date(dataset, end_date)
    dataset.drop_duplicates(inplace=True)

    # drop longest cases and debiasing end of dataset
    dataset_short, latest_start = limited_duration(dataset, max_days)

    targets = "remain_time"
    # create targets for classification
    if keywords_dict:
        dataset_short = createClassificationTargets(
            dataset_short, "classif_target", keywords_dict
        )
        dataset_short.drop(columns=["classif_target"], inplace=True)
        targets = list(keywords_dict.keys()) + ["remain_time"]

    # Split dataset in train and test set, applying strict temporal splitting and debiasing the test set
    dataset_train, dataset_test = trainTestSplit(
        dataset_short,
        test_len=test_len_share,
        latest_start=latest_start,
        targets=targets,
    )

    # record outputs
    if output_type == "xes":
        log_train = converter.apply(
            dataset_train, variant=converter.Variants.TO_EVENT_LOG
        )
        print("dataset_train converted to logs")
        xes_exporter.apply(log_train, path + "/" + file_name + "_train.xes")
        print("dataset_train exported as xes")
        log_test = converter.apply(
            dataset_test, variant=converter.Variants.TO_EVENT_LOG
        )
        print("dataset_test converted to logs")
        xes_exporter.apply(log_test, path + "/" + file_name + "_test.xes")
        print("dataset_test exported as xes")
    elif output_type == "pickle":
        dataset_train.to_pickle(path + "/" + file_name + "_train.pkl")
        dataset_test.to_pickle(path + "/" + file_name + "_test.pkl")
    elif output_type == "csv":
        dataset_train.to_csv(path + "/" + file_name + "_train.pkl")
        dataset_test.to_csv(path + "/" + file_name + "_test.pkl")
    else:
        print("output type unknown. Should be 'xes', 'pickle' or 'csv'")


def unbiased(
    dataset,
    path,
    file_name,
    start_date,
    end_date,
    max_days,
    test_len_share,
    output_type="xes",
    event_col="activity",
):
    """
    Creates benchmark training and test sets for next event problems

    Args:
        dataset: pandas DataFrame
        path: string: output file path
        file_name: string: output file name
        start_date: string "MM-YYYY": dataset starts here after removing outliers
        end_date: string "MM-YYYY": dataset stops here after removing outliers
        max_days: float
        test_len_share: float: share of cases belonging in test set
        output_type: string: "xes", "pickle" or "csv"
        event_col: string: column containing the event names
    Returns:
        two files ('_train' and '_test')
    """

    # remove chronological outliers and duplicates
    dataset["time:timestamp"] = pd.to_datetime(dataset["time:timestamp"], utc=True)
    if start_date:
        dataset = start_from_date(dataset, start_date)
    if end_date:
        dataset = end_before_date(dataset, end_date)
    dataset.drop_duplicates(inplace=True)

    # drop longest cases and debiasing end of dataset
    dataset_short, latest_start = limited_duration(dataset, max_days)

    # preliminaries
    case_stops_df = dataset.groupby("case:concept:name")["time:timestamp"].max()

    ### TRAINING AND TEST SET ###
    sorted_timestamps = np.sort(dataset["time:timestamp"].values)
    separation_time = sorted_timestamps[
        int(len(sorted_timestamps) * (1 - test_len_share))
    ]
    print(separation_time)

    # train_case_nrs = case_stops_df[case_stops_df.values <= separation_time].index.array
    dataset_train = dataset[
        dataset["time:timestamp"] <= pd.to_datetime(separation_time, utc=True)
    ].reset_index(drop=True)
    test_case_nrs = case_stops_df[case_stops_df.values > separation_time].index.array
    dataset_test = dataset[
        dataset["case:concept:name"].isin(test_case_nrs)
    ].reset_index(drop=True)

    # compute targets: next event
    def calcNextEvent(grp):
        grp["nextEvent"] = grp[event_col].shift(periods=-1)
        return grp

    dataset_train = dataset_train.groupby("case:concept:name").apply(calcNextEvent)
    dataset_train = dataset_train.dropna(subset=["nextEvent"]).reset_index(drop=True)
    dataset_test = dataset_test.groupby("case:concept:name").apply(calcNextEvent)
    dataset_test = dataset_test.dropna(subset=["nextEvent"]).reset_index(drop=True)

    # convert targets into np.NAN for those prefixes in test set
    # that end before the separation time (beginning of test set)
    dataset_test.loc[
        dataset_test["time:timestamp"].values < separation_time, "nextEvent"
    ] = np.nan

    # record outputs
    if output_type == "xes":
        log_train = converter.apply(
            dataset_train, variant=converter.Variants.TO_EVENT_LOG
        )
        print("dataset_train converted to logs")
        xes_exporter.apply(log_train, path + "/" + file_name + "_next_event_train.xes")
        print("dataset_train exported as xes")
        log_test = converter.apply(
            dataset_test, variant=converter.Variants.TO_EVENT_LOG
        )
        print("dataset_test converted to logs")
        xes_exporter.apply(log_test, path + "/" + file_name + "_next_event_test.xes")
        print("dataset_test exported as xes")
    elif output_type == "pickle":
        dataset_train.to_pickle(path + "/" + file_name + "_next_event_train.pkl")
        dataset_test.to_pickle(path + "/" + file_name + "_next_event_test.pkl")
    elif output_type == "csv":
        dataset_train.to_csv(path + "/" + file_name + "_next_event_train.pkl")
        dataset_test.to_csv(path + "/" + file_name + "_next_event_test.pkl")
    else:
        print("output type unknown. Should be 'xes', 'pickle' or 'csv'")
