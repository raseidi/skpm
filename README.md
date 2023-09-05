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
        ("time", TimestampExtractor(case_col="case_id", time_col="timestamp", features="execution_time")),
        ("scale", StandardScaler()),
    ]
)

# categorical transformer
cat_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
)

# column transformer (concating pipelines)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, ["activity"]),               # categorical features
        ("time", time_transformer, ["case_id", "timestamp"]), # case_id and timestamp
    ],
    remainder="drop",
).set_output(transform="pandas")

# complete learning pipeline 
reg = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestRegressor(n_jobs=-1))
    ]
)
reg.fit(log, log["resource"]) # resource as a toy example of numerical target
```