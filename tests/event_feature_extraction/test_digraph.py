import numpy as np
import pandas as pd
import pytest
from skpm.event_feature_extraction.meta import DigraphFeaturesExtractor
from skpm.config import EventLogConfig as elc


def test_digraph_features_extractor():
    # Test fit_transform with random data
    X = pd.DataFrame(
        {
            elc.case_id: np.random.randint(1, 10, 100),
            elc.activity: np.random.choice(["a", "b", "c"], 100),
        }
    )
    feature_extractor = DigraphFeaturesExtractor()
    transformed_data = feature_extractor.fit_transform(X)
    assert isinstance(transformed_data, pd.DataFrame)

    # Test transform without fit
    with pytest.raises(ValueError):
        feature_extractor = DigraphFeaturesExtractor()
        feature_extractor.transform(X)

    # Test with empty dataframe
    empty_X = pd.DataFrame(columns=[elc.case_id, elc.activity])
    feature_extractor = DigraphFeaturesExtractor()
    transformed_data = feature_extractor.fit_transform(empty_X)
    assert isinstance(transformed_data, pd.DataFrame)
    assert len(transformed_data) == 0

    # Additional tests can be added for specific features if needed
