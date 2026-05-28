"""Predict remaining time on the Sepsis event log with a vanilla sklearn Pipeline.

Run:
    uv run python examples/bpi20_rt.py
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skpm.config import EventLogConfig
from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.sequence_encoding import Aggregation


def main() -> None:
    # 1. Load the log (auto-downloads on first run).
    log = BPI20RequestForPayment(cache_folder="../ppm-llm/data/").dataframe

    # 2. Case-level holdout. split.temporal would put every case in train on
    #    this log because all cases start before the temporal cutoff.
    case_col = EventLogConfig.case_id
    train_cases, test_cases = train_test_split(
        log[case_col].unique(), test_size=0.2, random_state=0
    )
    train = log[log[case_col].isin(train_cases)].reset_index(drop=True)
    test = log[log[case_col].isin(test_cases)].reset_index(drop=True)

    # 3. Per-event remaining-time target (in hours).
    y_train = remaining_time(train, time_unit="h")
    y_test = remaining_time(test, time_unit="h")

    # 4. Pipeline: time features per event -> rolling-mean prefix encoding -> GBR.
    pipe = Pipeline(
        [
            (
                "time_features",
                TimestampExtractor(
                    case_features=None,
                    event_features="all",
                    targets=None,
                    time_unit="h",
                ),
            ),
            ("encode", Aggregation(method="mean")),
            ("model", HistGradientBoostingRegressor(random_state=1)),
        ]
    )

    pipe.fit(train, y_train)
    preds = pipe.predict(test)

    print(f"n_train events: {len(train):,}")
    print(f"n_test  events: {len(test):,}")
    print(f"MAE (hours):    {mean_absolute_error(y_test, preds):.2f}")


if __name__ == "__main__":
    main()
