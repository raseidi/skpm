import pandas as pd

from skpm.event_logs.base import LogLike, to_event_log


def _case_bounds(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-case (min, max) timestamp summary read from the event-log index."""
    timestamps = dataset.index.get_level_values("timestamp").to_series(
        index=dataset.index, name="timestamp"
    )
    return timestamps.groupby(level="case_id", sort=False, observed=True).agg(
        ["min", "max"]
    )


def _select_cases(dataset: pd.DataFrame, case_ids) -> pd.DataFrame:
    mask = dataset.index.get_level_values("case_id").isin(case_ids)
    return dataset[mask]


def _check_test_size(test_size: float) -> None:
    """Reject sizes that would index out of range or wrap around.

    Lives in the primitives so every caller inherits it. Without it,
    ``test_size=1.5`` makes :func:`unbiased` index ``bounds`` with a negative
    position and return a silently wrong split.
    """
    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be a fraction strictly between 0 and 1, got "
            f"{test_size!r}."
        )


def _check_nonempty_split(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> None:
    """Guard against the silent empty-side footgun."""
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError(
            f"Split produced an empty side (train={len(df_train)} events, "
            f"test={len(df_test)} events). This usually means every case falls "
            f"on one side of the cutoff; adjust test_size / date bounds, or use a "
            f"case-level holdout."
        )


def _bounded_dataset(
    dataset: pd.DataFrame, start_date, end_date
) -> pd.DataFrame:
    bounds = _case_bounds(dataset)
    timestamps = dataset.index.get_level_values("timestamp")

    # Drop tz before to_period — we only need the calendar month, and this
    # avoids pandas' "Converting to PeriodArray will drop timezone" warning.
    start = (
        pd.Period(start_date)
        if start_date
        else timestamps.min().tz_localize(None).to_period("M")
    )
    end = (
        pd.Period(end_date)
        if end_date
        else timestamps.max().tz_localize(None).to_period("M")
    )

    keep = (bounds["min"].dt.tz_localize(None).dt.to_period("M") >= start) & (
        bounds["max"].dt.tz_localize(None).dt.to_period("M") <= end
    )
    return _select_cases(dataset, bounds.index[keep])


def _unbiased(dataset: pd.DataFrame, max_days: float) -> pd.DataFrame:
    bounds = _case_bounds(dataset).assign(
        duration=lambda x: (x["max"] - x["min"]).dt.total_seconds()
        / (24 * 60 * 60)
    )

    condition_1 = bounds["duration"] <= max_days * 1.00000000001
    latest_start = dataset.index.get_level_values(
        "timestamp"
    ).max() - pd.Timedelta(max_days, unit="D")
    condition_2 = bounds["min"] <= latest_start

    keep = condition_1 & condition_2
    return _select_cases(dataset, bounds.index[keep])


def unbiased(
    dataset: LogLike,
    start_date: str | pd.Period | None,
    end_date: str | pd.Period | None,
    max_days: float,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unbiased split of an event log into training and test sets [1]_.

    Prefer :func:`skpm.model_selection.train_test_split`, which resolves the
    per-dataset parameters below from the loader. This is the primitive it
    calls: it takes its parameters and nothing else.

    The event log is expected in canonical form (
    :func:`skpm.event_logs.base.to_event_log`); raw DataFrames are
    promoted on entry.

    Parameters
    ----------
    dataset : pd.DataFrame or EventLog
    start_date, end_date : str or pd.Period or None
        Optional bounds on the case start/end (monthly resolution).
    max_days : float
        Maximum allowed case duration in days.
    test_size : float, default=0.2
        Proportion of cases to use for the test set.

    References
    ----------
    .. [1] Hans Weytjens, Jochen De Weerdt. Creating Unbiased Public
       Benchmark Datasets with Data Leakage Prevention for Predictive
       Process Monitoring, 2021. doi:10.1007/978-3-030-94343-1_2
    """
    _check_test_size(test_size)
    dataset = to_event_log(dataset).copy()

    if start_date or end_date:
        dataset = _bounded_dataset(dataset, start_date, end_date)
    dataset = _unbiased(dataset, max_days)

    bounds = _case_bounds(dataset)
    if len(bounds) == 0:
        raise ValueError(
            f"The unbiased filters left no cases to split: every case either "
            f"lasts longer than max_days={max_days} or starts within the last "
            f"max_days of the log. Raise max_days, or widen "
            f"start_date/end_date."
        )

    first_test_case_nr = int(len(bounds) * (1 - test_size))
    first_test_start_time = (
        bounds["min"].sort_values().values[first_test_case_nr]
    )
    test_cases = bounds.index[bounds["max"].values >= first_test_start_time]

    df_test = _select_cases(dataset, test_cases)
    df_train = _select_cases(dataset, bounds.index.difference(test_cases))
    _check_nonempty_split(df_train, df_test)
    return df_train, df_test


def temporal(
    dataset: LogLike, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: any case whose first event is at or before the
    cutoff goes to train, the rest to test.

    Prefer :func:`skpm.model_selection.train_test_split`, the documented door;
    this is the primitive it calls.

    Note that a train case keeps *all* of its events, including any dated
    after the cutoff. Whole cases stay on one side, which is what keeps the
    case-local targets in :mod:`skpm.feature_extraction.targets` well defined
    on each side; it also means the training set can reach past the cutoff.
    Use :func:`unbiased` for a protocol that controls this.

    ``test_size`` is a fraction of the log's **calendar span**, not of its
    cases: the cutoff sits that far through the time axis, so uneven case
    arrivals give correspondingly uneven case counts.
    """
    _check_test_size(test_size)
    dataset = to_event_log(dataset)
    timestamps = dataset.index.get_level_values("timestamp")
    start, end = timestamps.min(), timestamps.max()
    split_point = start + (end - start) * (1 - test_size)

    train_mask_per_event = (
        dataset.index.get_level_values("timestamp") <= split_point
    )
    case_ids = dataset.index.get_level_values("case_id")
    train_cases = case_ids[train_mask_per_event].unique()

    df_train = _select_cases(dataset, train_cases)
    df_test = dataset[
        ~dataset.index.get_level_values("case_id").isin(train_cases)
    ]
    _check_nonempty_split(df_train, df_test)
    return df_train, df_test
