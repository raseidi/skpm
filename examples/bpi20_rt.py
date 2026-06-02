"""Predict remaining time on a BPI20 event log with a vanilla sklearn Pipeline.

Run:
    uv run python examples/bpi20_rt.py
"""

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.sequence_encoding import Aggregation


def main() -> None:
    # 1. Load the log (auto-downloads on first run). The dataframe carries
    #    the canonical event-log MultiIndex (case_id, timestamp, event_id).
    bpi = BPI20RequestForPayment(cache_folder="../ppm-llm/data/")
    log = bpi.dataframe

    # 2. Case-level holdout. split.temporal would put every case in train on
    #    this log because all cases start before the temporal cutoff.
    case_ids = log.index.get_level_values("case_id").unique()
    train_cases, test_cases = train_test_split(
        case_ids, test_size=0.2, random_state=0
    )
    train = log[log.index.get_level_values("case_id").isin(train_cases)]
    test = log[log.index.get_level_values("case_id").isin(test_cases)]

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
            ("model", RandomForestRegressor(random_state=1, n_jobs=-1)),
        ]
    )

    pipe.fit(train, y_train)
    preds = pipe.predict(test)

    print(f"n_train events: {len(train):,}")
    print(f"n_test  events: {len(test):,}")
    print(f"MAE (hours):    {mean_absolute_error(y_test, preds):.2f}")


if __name__ == "__main__":
    main()
