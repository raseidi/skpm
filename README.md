# Scikit-PM

"scikit"nizing ml pipelines for process mining. **Under development**.

---

## Usage

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skpm.event_feature_extraction import TimestampExtractor

# timestamp transformer
time_transformer = Pipeline(
    steps=[
        ("time", TimestampExtractor(case_col="case_id", time_col="timestamp", features="all")),
        ("scale", StandardScaler()),
    ]
)

# activity encoding
cat_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
)

# preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("oh", cat_transformer, ["activity"]),
        ("time", time_transformer, ["case_id", "timestamp"]),
        ("case_id", "passthrough", ["case_id"]), 
    ],
    remainder="passthrough",
).set_output(transform="pandas")

# classification pipeline
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("aggregator", TraceAggregator(case_col="case_id", method="mean")),
        ("classifier", RandomForestClassifier(n_jobs=-1))
    ]
).set_output(transform="pandas")

# running pipeline
clf.fit(log_train, log_train)
print(clf.score(log_test, log_test))
```