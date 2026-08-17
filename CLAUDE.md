# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run tests with coverage
uv run pytest --cov=skpm tests

# Run a single test file
uv run pytest tests/test_<module>.py

# Format code
black src tests

# Type checking
uv run mypy src
```

## Architecture

SkPM is a scikit-learn extension for Process Mining. It wraps sklearn's `BaseEstimator`/`TransformerMixin` APIs with process-mining–specific conventions, primarily around event log DataFrames.

### Data shape contract (`event_logs/base.py`)

Every event-log DataFrame inside skpm carries the canonical **MultiIndex** with levels `("case_id", "timestamp", "event_id")`:

- `case_id` — the per-case identifier.
- `timestamp` — UTC datetime of the event. Kept as a key (not a column) so it survives downstream feature extraction; users still see it in `df.index.get_level_values("timestamp")` for post-hoc analysis (e.g. error-by-day plots).
- `event_id` — 0-based row counter assigned by `to_event_log` after a stable sort by `(case_id, timestamp)`. Globally unique across the log, so simultaneous events (same case, same timestamp) stay distinguishable in original document order.

The level names are **semantic constants** independent of how source columns are named. `EventLogConfig` governs the *source-column* names; the index level names are fixed.

The canonical shape is set by `skpm.event_logs.base.to_event_log(log, column_mapping=...)`, which:

1. normalizes column names via `EventLogConfig.normalize_columns`,
2. parses timestamps to UTC,
3. sorts stably by `(case_id, timestamp)`,
4. assigns `event_id` as a 0-based row counter (`df.index.values` after the stable sort),
5. moves `case_id` and `timestamp` into the MultiIndex.

**One entry-point convention: `to_event_log` is the single coercion boundary, and every skpm API that takes a log accepts all three log-like inputs** — an `EventLog` (any `skpm.event_logs` loader), a flat DataFrame, or an already-canonical DataFrame. The alias `LogLike` (exported from `skpm`, defined in `event_logs/base.py`) names that union. Estimators (`_validate_log`), targets, splits, and the index accessors all route through it, so `.dataframe` is available but never required:

```python
TimestampExtractor().fit(BPI17())            # EventLog
remaining_time(BPI17())                      # same object, different API
Pipeline([("time", TimestampExtractor()), ("m", RandomForestRegressor())]).fit(BPI17(), y)
```

Two guards live at that boundary: `column_mapping` combined with an `EventLog` raises (the loader already normalized, so it would be a silent no-op), and an `EventLog` holding no data raises a named error instead of `AttributeError`.

This covers **skpm** estimators only. A plain sklearn estimator still needs a feature matrix — a raw log has string activities and nothing encoded — so `RandomForestRegressor().fit(BPI17())` fails by design. Put a skpm transformer in front of it, which is what a Pipeline does. Don't "fix" this by duck-typing `EventLog` as a DataFrame.

**The `EventLog`-as-input convenience stops at sklearn's door, and this is a documented carve-out, not a bug to fix.** sklearn meta-estimators (`GridSearchCV`, `cross_val_score`, `cross_validate`) call `indexable(X, y, groups)` and `_safe_indexing` *before* any skpm code runs, and an `EventLog` satisfies neither — it has no `iloc`, `__len__`, or `shape`, so `indexable` raises `TypeError: Input should have at least 1 dimension`. Cross-validation therefore requires a DataFrame. That is a second reason `to_event_log` must keep returning a DataFrame, and a reason the documented workflow splits first: after a split you hold frames. Treat estimator-level `EventLog` acceptance as a convenience for exploratory one-liners, not the headline workflow.

`EventLog.__init__` calls `to_event_log` on construction.

**Public index accessors** (top-level `skpm.case_ids`, `skpm.timestamps`, `skpm.event_ids`, `skpm.trace_positions`, plus `skpm.to_event_log`) pull an index level out as a Series aligned to the event log. Use these instead of reaching into `df.index.get_level_values(...)` directly.

### Splitting (`model_selection.py`)

**Splitting comes first in a skpm workflow** — before feature extraction, before fitting anything. `skpm.model_selection.train_test_split(log, strategy=...)` is the documented door (also exported as `skpm.train_test_split`); `temporal` and `unbiased` in `event_logs/split.py` stay public as the primitives it dispatches to, and take their parameters and nothing else.

```python
train, test = train_test_split(BPI17(), strategy="unbiased")   # params from the loader
train, test = train_test_split(pd.read_csv("mine.csv"))        # temporal, the default
y_train = remaining_time(train, time_unit="h")                 # target, per side
```

Three deliberate properties:

- **It returns two event logs, never `(X_train, X_test, y_train, y_test)`.** Which target to predict is a modelling decision (remaining time is regression, next activity is classification), and the same split also serves unsupervised work. A 4-tuple would force a `target=` argument and then absorb every target's options. It would also buy nothing: both strategies are **case-level** and all three targets in `feature_extraction/targets.py` are case-local groupbys, so computing a target before or after the split gives identical values — pinned by `tests/test_model_selection.py::test_target_is_unchanged_by_splitting_after_instead_of_before`. The leakage that is real lives on the X side (`WorkInProgress` is inter-case; `Aggregation`/`ResourcePoolExtractor`/`Bucketing` are fitted), and only fitting the Pipeline on train alone prevents it.
- **Naming them `train`/`test` rather than `X_train`/`X_test` is intentional.** They are raw event logs with string activities, not feature matrices; `X_` would teach exactly the mental model the carve-out above warns against.
- **The unbiased parameters are resolved from the loader, never guessed.** `start_date`/`end_date`/`max_days` are per-dataset constants published with the benchmark (Weytjens & De Weerdt) and only 6 of the 14 loaders ship them. Precedence is explicit kwarg → loader → error naming what's missing. This needs the one documented `isinstance(log, EventLog)` outside the coercion boundary, in `_loader_params`, because `to_event_log` erases the distinction. Passing an unbiased-only parameter with `strategy="temporal"` raises rather than silently ignoring it.

Subclasses that publish parameters set the **private** `_unbiased_split_params`; the public `TUEventLog.unbiased_split_params` property validates and returns a **copy**. (Setting the public name directly shadows the property — that was a live bug: the property's return path was unreachable and all instances shared one mutable dict.)

### Column naming (`config.py`)

The canonical field names — `case_id`, `timestamp`, `activity`, `resource` — are **fixed constants**, not configurable; they are how skpm addresses columns internally (`case_id`/`timestamp` → index levels, `activity`/`resource` → columns), and are exposed as the `.case_id`/`.timestamp`/`.activity`/`.resource` properties. What *is* configurable is the expected **source** column name per field — one name each, defaulting to XES (`case:concept:name`, `time:timestamp`, `concept:name`, `org:resource`). `normalize_columns(df, mapping=None)` resolves each field by precedence — explicit `mapping` > configured source name (if present) > already-canonical column (if present) — then renames it to the canonical name. There is **no alias guessing**: declare non-default naming globally via `set_global_config(case_id="CaseID")` or per-load via `column_mapping={"case_id": "CaseID"}`. Required fields `case_id`/`timestamp`/`activity` raise if unresolved; `resource` is optional. Because the canonical names are constants, estimators read them directly from `EventLogConfig` — there is no fit-time config snapshot.

### Base classes (`base.py`)

- `BaseProcessEstimator` — extends `BaseEstimator + EventLogConfigMixin`. `_validate_log()` enforces the canonical event-log shape, delegating coercion to `to_event_log` (so it unwraps an `EventLog` and promotes a flat frame; polars raises `NotImplementedError` — temporarily unsupported). Index helpers return a Series aligned to `X.index`: `_case_ids`, `_timestamps`, `_event_ids` (the global per-event counter), and `_trace_positions` (0-based position within each case).
- `BaseProcessTransformer` — the **event-level** (row-preserving) base, extending `TransformerMixin + BaseProcessEstimator`. Subclasses implement `_fit(X, y)` and `_transform(X, y)` and **must not** override `fit`/`transform`: the base validates/promotes the log (so `_fit`/`_transform` always receive a canonical frame), sets the `fitted_` marker, and checks that DataFrame output columns match `get_feature_names_out()`.
- `CaseLevelTransformer` — base for **trace-level** transformers that emit one row per case (e.g. `VariantExtractor`). Cardinality is declared via the `_cardinality` class attribute (`"event"` vs `"case"`). It is a terminal step, not a Pipeline intermediate — this explicit contract replaced the old `ensure_not_pipeline` stack-inspection guard.

**Note on MultiIndex preservation:** every event-level `_transform` must index its own output by `X.index`. Do not return a bare ndarray and rely on sklearn's `set_output` wrapper to supply the index: it recovers one only when the object handed to `transform` is *itself* a DataFrame (it deliberately avoids `getattr(x, "index")`), so an `EventLog` caller would silently get a flat index. With every transformer indexing its own output, the canonical MultiIndex now propagates end to end for all three input forms — `EventLog`, canonical, and flat. `tests/test_base.py::test_event_log_input_matches_dataframe_input` pins this.

### Main modules

| Module | Purpose |
|---|---|
| `feature_extraction/` | Event-level feature transformers: `TimestampExtractor` (temporal features), `ResourcePoolExtractor` (resource roles), `WorkInProgress` (inter-case concurrency). `TimestampExtractor` exposes **past-looking features only**: each feature class declares an explicit `FEATURES` registry tuple (`"all"` resolves to exactly that tuple; unknown names raise), and `TimestampCaseLevel.TARGET_ONLY` blocks future-looking computations (`execution_time`, `remaining_time`) from feature selection. To add a feature: add the classmethod *and* list it in `FEATURES`. `targets.py` holds `remaining_time` / `execution_time` / `next_activity` — **functions** (not transformers; targets are label generation, which sklearn keeps outside the X-pipeline) returning a 1-D Series aligned to the event-log index, to pass as `y` to `pipe.fit(X, y)` |
| `sequence_encoding/` | Prefix encoders that turn each event's prefix into a fixed-length feature vector: `Aggregation` (order-agnostic summary stats), `Indexing` (absolute index-based encoding — full prefix width, no future leakage), `Windowing` (relative sliding window over the most recent `n` events), `Bucketing`. `Indexing`/`Windowing` share the private `_PositionalEncoder` base (`index.py`/`windowing.py`/`_positional.py`) and zero-pad structurally-missing cells by default so output is NaN-free. |
| `event_logs/` | Download BPI Challenge logs from the 4TU repository (`bpi.py`), parse XES/CSV event logs (`parser.py`), `split.py` split primitives (`temporal`, `unbiased`) which raise on an empty train/test side — reached via `skpm.model_selection.train_test_split` |
| `model_selection.py` | `train_test_split(log, strategy="temporal"\|"unbiased", test_size=..., ...)` — the documented splitting door; resolves the unbiased parameters from a loader and re-exports the two primitives |
| `utils/` | Validation helpers (`validation.py`), time utilities, graph helpers |

### sklearn integration

The package sets `sklearn.set_config(transform_output="pandas")` globally at import time, so all transformers return pandas DataFrames. Transformers plug directly into `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.

### pandas/polars

Polars support is **temporarily disabled**: `_validate_log` raises `NotImplementedError` on polars input. `Aggregation` still carries an `engine="polars"` parameter and a `_transform_polars` implementation (kept for re-enablement), and its two polars tests are skipped. The pandas path is the only supported route today; outputs are always pandas DataFrames. Re-enabling polars is tracked as future work.

## CI

GitHub Actions runs `uv run pytest --cov=skpm tests` on Python 3.11 and 3.12 (ubuntu-latest) on every push/PR to `main`.
