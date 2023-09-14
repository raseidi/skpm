# Scikit-PM

"Scikit"nizing ML pipelines for process mining. **Under development**.

---

## Installation

Create a local enviroment using `conda` and install the `requirements.txt`.

Conda:

```bash
conda create --name skpm python=3.10
pip install -r requirements.txt
```

**NOTE**: Soon to be on pip.

---

## Tutorials

Please, check out a few use cases in our `use_cases/` directory:

- [Event feature extraction](use_cases/time_features.ipynb)
- [Implementing ML pipelines](use_cases/pipeline.ipynb)
- [Tuning and model selection](.) (under dev)

---

## Basic usage

Below, a quick example of how extracting timestamp-realted features:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skpm.event_feature_extraction import TimestampExtractor

# timestamp transformer
tt = TimestampExtractor(
    features="all"
).set_output(transformer="pandas") # output as pandas dataframe

# running pipeline
out = tt.fit_transform(log)
print(out)
```