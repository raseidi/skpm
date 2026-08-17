# SkPM

*Process mining, the scikit-learn way.*

[![Read the Docs](https://img.shields.io/readthedocs/skpm)](https://skpm.readthedocs.io/en/latest/)
[![Codecov](https://img.shields.io/codecov/c/github/raseidi/skpm)](https://codecov.io/gh/raseidi/skpm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SkPM** is a Python module for process mining built on top of scikit-learn and
pandas, and distributed under the MIT license.

Event logs are not tables of independent rows: cases unfold over time, every
event is a moment where a prediction could be made, and almost every convenient
shortcut leaks the future into the past. SkPM turns an event log into a
supervised learning problem without leaking it — time-aware splits, prediction
targets, feature extraction and prefix encoding, all as ordinary scikit-learn
transformers, so they compose with the `Pipeline`, `ColumnTransformer` and
`GridSearchCV` you already use.

Website: https://skpm.readthedocs.io

## Installation

### User installation

A PyPI release is on the way. Until then, install the latest version from the
repository with `uv` or `pip`:

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install git+https://github.com/raseidi/skpm.git
```

## Getting started

How much longer does each running case need? On a public benchmark log:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from skpm.event_logs import BPI20RequestForPayment
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.model_selection import train_test_split
from skpm.sequence_encoding import Aggregation

# Payment requests from a Dutch university; downloads and caches on first use.
log = BPI20RequestForPayment()

# Split first, before extracting features: whole cases, separated in time.
train, test = train_test_split(log, strategy="unbiased")

# Hours until each case finishes. Which target to predict is your choice.
y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

# Fitted on the training cases only, which is what keeps the estimate honest.
pipeline = Pipeline([
    ("features", TimestampExtractor(time_unit="h")),
    ("prefix", Aggregation(method="mean")),
    ("model", HistGradientBoostingRegressor()),
]).fit(train, y_train)

print(mean_absolute_error(y_test, pipeline.predict(test)))
```

Your own log works the same way, with no loader class and no conversion step to
remember: `train_test_split(pd.read_csv("my_log.csv"))`.

## Documentation

- [Quickstart](https://skpm.readthedocs.io/en/latest/auto_examples/plot_quickstart.html) — the whole workflow on one page, on synthetic data.
- [User Guide](https://skpm.readthedocs.io/en/latest/user_guide/index.html) — the event log, splitting, targets, features, prefix encoding, composition.
- [Examples](https://skpm.readthedocs.io/en/latest/auto_examples/index.html) — next-activity prediction, the prefix encodings side by side, and a real BPI Challenge log end to end.
- [API reference](https://skpm.readthedocs.io/en/latest/autoapi/index.html) — every public class and function.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to
open an issue or a pull request. SkPM follows scikit-learn's `fit` / `transform`
/ `predict` conventions, so its
[contributing guide](https://scikit-learn.org/stable/developers/contributing.html)
applies here too.

## Citation

SkPM was presented at the CoopIS 2023 demonstration track.

```bibtex
@inproceedings{OyamadaTJC23,
  author    = {Rafael Seidi Oyamada and
               Gabriel Marques Tavares and
               Sylvio Barbon Junior and
               Paolo Ceravolo},
  title     = {A Scikit-learn Extension Dedicated to Process Mining Purposes},
  booktitle = {Proceedings of the Demonstration Track co-located with the
               International Conference on Cooperative Information Systems 2023},
  series    = {{CEUR} Workshop Proceedings},
  publisher = {CEUR-WS.org},
  year      = {2023},
}
```
