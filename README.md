# Scikit-PM

"scikit"nizing event log preprocessing

---

## Usage

```python
# timestamp transformer
time_transformer = Pipeline(
    steps=[
        ("time", Timestamp(case_col="case_id", time_col="timestamp", features="execution_time")),
        ("scale", StandardScaler()),
    ]
)

# categorical transformer
cat_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
)

# column transformer (embedding pipelines)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, ["activity"]),               # categorical features
        ("time", time_transformer, ["case_id", "timestamp"]), # case_id and timestamp
    ],
    remainder="drop",
).set_output(transform="pandas")

# complete learning pipeline 
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestRegressor(n_jobs=-1))
    ]
)
clf.fit(log, log["resource"]) # resource as a toy example of numerical target
```