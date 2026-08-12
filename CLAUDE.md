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

**One entry-point convention: `to_event_log` is the single coercion boundary, and every skpm API that takes a log accepts all three log-like inputs** — an `EventLog` (any `skpm.event_logs` loader), a flat DataFrame, or an already-canonical DataFrame. The alias `LogLike` in `event_logs/base.py` names that union. Estimators (`_validate_log`), targets, splits, and the index accessors all route through it, so `.dataframe` is available but never required:

```python
TimestampExtractor().fit(BPI17())            # EventLog
remaining_time(BPI17())                      # same object, different API
Pipeline([("time", TimestampExtractor()), ("m", RandomForestRegressor())]).fit(BPI17(), y)
```

Two guards live at that boundary: `column_mapping` combined with an `EventLog` raises (the loader already normalized, so it would be a silent no-op), and an `EventLog` holding no data raises a named error instead of `AttributeError`.

This covers **skpm** estimators only. A plain sklearn estimator still needs a feature matrix — a raw log has string activities and nothing encoded — so `RandomForestRegressor().fit(BPI17())` fails by design. Put a skpm transformer in front of it, which is what a Pipeline does. Don't "fix" this by duck-typing `EventLog` as a DataFrame.

`EventLog.__init__` calls `to_event_log` on construction.

**Public index accessors** (top-level `skpm.case_ids`, `skpm.timestamps`, `skpm.event_ids`, `skpm.trace_positions`, plus `skpm.to_event_log`) pull an index level out as a Series aligned to the event log. Use these instead of reaching into `df.index.get_level_values(...)` directly.

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
| `event_logs/` | Download BPI Challenge logs from the 4TU repository (`bpi.py`), parse XES/CSV event logs (`parser.py`), `split.py` train/test splits (`temporal`, `unbiased`) which raise on an empty train/test side |
| `utils/` | Validation helpers (`validation.py`), time utilities, graph helpers |

### sklearn integration

The package sets `sklearn.set_config(transform_output="pandas")` globally at import time, so all transformers return pandas DataFrames. Transformers plug directly into `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.

### pandas/polars

Polars support is **temporarily disabled**: `_validate_log` raises `NotImplementedError` on polars input. `Aggregation` still carries an `engine="polars"` parameter and a `_transform_polars` implementation (kept for re-enablement), and its two polars tests are skipped. The pandas path is the only supported route today; outputs are always pandas DataFrames. Re-enabling polars is tracked as future work.

## CI

GitHub Actions runs `uv run pytest --cov=skpm tests` on Python 3.10 and 3.12 (ubuntu-latest) on every push/PR to `main`.
