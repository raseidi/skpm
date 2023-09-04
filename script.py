import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skpm.event_feature_extraction import TimeStampExtractor
from skpm.preprocessing import TraceAggregator

log = pd.read_csv("data/bpi2012.csv")
log = log.rename(
    columns={
        "case:concept:name": "case_id",
        "time:timestamp": "timestamp",
        "concept:name": "activity",
        "org:resource": "resource",
    }
)
log = log.loc[:, ["case_id", "activity", "timestamp", "resource"]]
log["case_id"] = log["case_id"].astype("category")
log["timestamp"] = pd.to_datetime(log["timestamp"])
log.resource = log.resource.fillna(0)

# sc = StandardScaler(with_mean=True).set_output(transform="pandas")
# sc.fit_transform(log[["resource"]])

t = TimeStampExtractor(case_col="case_id", time_col="timestamp")
t.fit(log)
t.get_feature_names_out()
features = t.fit_transform(log)
# log = t.transform(log)

# trace_encoder = OneHotEncoder(sparse_output=False)
# trace_encoder.fit(log[["activity"]]).set_output(transform="pandas")
# log = pd.concat((log, trace_encoder.transform(log[["activity"]])), axis=1).drop(
#     columns=["activity"]
# )

# features_ = log.select_dtypes(exclude=["object"]).columns.values[2:]

# start = time.time()
# a = log.groupby("case_id", group_keys=False, as_index=False)[features_].transform(
#     lambda x: np.cumsum(x) / (1 + np.arange(len(x)))
# )
# print("First approach: ", time.time() - start)


# fn = lambda x: np.cumsum(x, axis=0) / (1 + np.arange(len(x)))
# start = time.time()
# for case in log["case_id"].unique():
#     mask = log["case_id"] == case
#     log.loc[mask, features_] = fn(log.loc[mask, features_].values)
# print("Second approach: ", time.time() - start)

# x = np.arange(9).reshape(3, 3)
# np.cumsum(x, axis=0)

# # trace_aggregator = TraceAggregator()
# # trace_aggregator.fit(log)
# # a = trace_aggregator.transform(log)
# # log[log.case_id == 173688]