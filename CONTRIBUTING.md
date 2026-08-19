# Contributing to SkPM

Thanks for your interest in SkPM. Bug reports, documentation fixes and pull
requests are all welcome, and credit is always given.

SkPM follows scikit-learn's conventions closely, so scikit-learn's
[contributing guide](https://scikit-learn.org/stable/developers/contributing.html)
is worth reading alongside this one — most of it applies here directly.

## Ways to contribute

- **Report a bug** or **request a feature** by opening an
  [issue](https://github.com/raseidi/skpm/issues).
- **Improve the documentation** — docstrings, the user guide, or an example for
  the gallery. This is the easiest place to start and always useful.
- **Contribute code** — fix a bug, or implement something discussed in an issue.

If you are looking for something to pick up, issues labelled `good first issue`
or `help wanted` are a good place to start.

## Submitting a bug report or a feature request

Before opening an issue, please search the existing ones — it may already be
known.

A good bug report includes:

- a **minimal reproducible example**: the shortest code that shows the problem,
  ideally on a small synthetic log rather than a downloaded BPI Challenge log,
  so it runs anywhere in seconds;
- the **full traceback**, not just the last line;
- your versions:

  ```console
  python -c "import skpm, sklearn, pandas; print(skpm.__version__, sklearn.__version__, pandas.__version__)"
  ```

For a feature request, describe the process-mining problem you are trying to
solve, not only the API you have in mind — there may be a simpler way to get
there with what already exists.

## Setting up a development environment

SkPM uses [uv](https://docs.astral.sh/uv/).

```console
git clone https://github.com/raseidi/skpm.git
cd skpm
uv sync                                   # project + dev tools
uv run pytest --cov=skpm tests            # run the suite
```

Some tests download small event logs from the 4TU repository, so the first run
needs a network connection.

For the documentation:

```console
uv sync --group docs
uv run sphinx-build docs/source docs/build
```

## Contributing code

1. Fork the repository and create a branch:

   ```console
   git checkout -b my-fix-or-feature
   ```

2. Make your change, with tests.
3. Run the suite and the formatter:

   ```console
   uv run pytest tests
   uv run black src tests
   ```

4. Push and open a pull request against `main`, describing what changed and why.

Keep pull requests focused on one thing — a small, self-contained change is far
easier to review and merge than a large one touching several areas.

## Pull request checklist

- [ ] Tests cover the new behaviour, and the full suite passes.
- [ ] Public functions and classes have docstrings, in the same style as the
      surrounding code.
- [ ] `black src tests` has been run.
- [ ] The user guide or an example is updated, if the change affects how SkPM is
      used.

Continuous integration runs the suite on Python 3.11 and 3.12; a green check is
required before merging.

## Coding conventions

SkPM is a scikit-learn extension, so estimators follow scikit-learn's API —
`fit` / `transform` / `predict`, parameters stored unmodified in `__init__`, and
everything learned from data stored in attributes ending with an underscore.

A few conventions are specific to SkPM, and are worth knowing before writing a
new transformer:

- **Event logs carry a MultiIndex** of `(case_id, timestamp, event_id)`.
  `skpm.event_logs.base.to_event_log` is the single place where a log is coerced
  into that shape.
- **Event-level transformers** subclass `BaseProcessTransformer` and implement
  `_fit` / `_transform`. Do not override `fit` / `transform` — the base class
  validates the log and checks the output columns for you.
- **Every `_transform` must index its output by `X.index`**, so the MultiIndex
  survives the whole pipeline.
- **Trace-level transformers** — one row per case — subclass
  `CaseLevelTransformer` instead.
- **Prediction targets are functions, not transformers** (see
  `feature_extraction/targets.py`): they return a Series aligned to the log, to
  be passed as `y`.
- **Features must not look into the future.** A feature computed from events
  that have not happened yet at prediction time is leakage, and is the single
  most common mistake in process-mining pipelines.

There is more detail in the [user guide](https://skpm.readthedocs.io/), and
`CLAUDE.md` in the repository root documents the internal contracts in depth.

## Code of Conduct

By participating in this project you agree to abide by its
[Code of Conduct](https://github.com/raseidi/skpm/blob/main/CONDUCT.md).
