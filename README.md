# SkPM: a Scikit-learn Extension for Process Mining

<p align="center">
  <img src="docs/logo.png" width="350"/>
</p>


<div align="center">

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Read the Docs](https://img.shields.io/readthedocs/skpm)](https://skpm.readthedocs.io/en/latest/)
[![Codecov](https://img.shields.io/codecov/c/github/raseidi/skpm)](https://codecov.io/gh/raseidi/skpm)

</div>

## Overview

SkPM is an open-source extension of the widely used [Scikit-learn](https://scikit-learn.org/) library, designed to meet the specific needs of Process Mining applications. It aims to provide a **standard**, **reproducible**, and **easily accessible** set of tools for PM research and practical applications.

## Available examples

- **NEW** [**ICPM/ML4PM 2024 Tutorial**](https://colab.research.google.com/drive/1s6TxG14bKbh2zlOENLGGd9dy_1BLEBiO?usp=sharing): A notebook highlighting all the available features in SkPM!
- [**Quickstart**](https://skpm.readthedocs.io/en/latest/auto_examples/plot_quickstart.html): Load an event log and extract past-looking temporal features.
- [**Select a remaining-time model**](https://skpm.readthedocs.io/en/latest/auto_examples/plot_tiny_benchmark.html): The end-to-end pattern — download a public log, split it (temporal or unbiased), define a target, and compare models with grouped cross-validation.
- [**Remaining time on BPI20**](https://skpm.readthedocs.io/en/latest/auto_examples/bpi20_rt.html): SkPM transformers inside a plain scikit-learn pipeline.

<p align="center">
  <img src="docs/pipeline.png"/>
</p>

## Installation

**Soon available on PyPI**.

To install SkPM, you can clone the repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/raseidi/skpm.git
cd skpm
pip install .
```

## Usage

Below is an example of how to use SkPM to build a pipeline for remaining time prediction.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skpm.event_logs import BPI17
from skpm.feature_extraction import TimestampExtractor
from skpm.feature_extraction.targets import remaining_time
from skpm.model_selection import train_test_split
from skpm.sequence_encoding import Aggregation

# 1. Split the log first — before extracting features or fitting anything.
#    Public logs ship the published unbiased-split parameters, so the strategy
#    reads them from the loader. Your own log works the same way:
#    train, test = train_test_split(pd.read_csv("my_log.csv"))
log = BPI17()
train, test = train_test_split(log, strategy="unbiased")

# 2. Choose the target. Remaining time is a regression; next_activity would be
#    a classification over the same two logs. Build it once per side.
y_train = remaining_time(train, time_unit="h")
y_test = remaining_time(test, time_unit="h")

# 3. Compose SkPM transformers with any sklearn estimator.
pipeline = Pipeline(steps=[
    ("timestamps", TimestampExtractor(time_unit="h")),
    ("prefix_encoding", Aggregation(method="mean")),
    ("standardization", StandardScaler()),
    ("regressor", RandomForestRegressor()),
])

# 4. Fit on the training cases only — this is what prevents leakage.
pipeline.fit(train, y_train)

predictions = pipeline.predict(test)
```

## Documentation

Detailed documentation and examples can be found [here](https://skpm.readthedocs.io/en/latest/).

## Roadmap, next steps, and help needed!

- Improving documentation by including examples.
- Implementing new applications and writing tutorials.
- Adding new methods (feature extraction, trace encoding, and models).
- Writing unit tests!

## Contributing

We welcome contributions from the community! 

Check the [sklearn guidelines](https://scikit-learn.org/1.5/developers/contributing.html#reading-the-existing-code-base) to understand the `fit`, `predict`, and `transform` APIs!

Check [our guidelines](CONTRIBUTING.md) as well to see how to open an issue or a PR. In summary:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'feat: add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project was created by Rafael Oyamada and is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). Feel free to use, modify, and distribute the code with attribution.

## Credits

`skpm` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## Citation

```bibtex
@inproceedings{OyamadaTJC23,
  author       = {Rafael Seidi Oyamada and
                  Gabriel Marques Tavares and
                  Sylvio Barbon Junior and
                  Paolo Ceravolo},
  editor       = {Felix Mannhardt and
                  Nour Assy},
  title        = {A Scikit-learn Extension Dedicated to Process Mining Purposes},
  booktitle    = {Proceedings of the Demonstration Track co-located with the International
                  Conference on Cooperative Information Systems 2023, CoopIS 2023, Groningen,
                  The Netherlands, October 30 - November 3, 2023},
  series       = {{CEUR} Workshop Proceedings},
  publisher    = {CEUR-WS.org},
}
```