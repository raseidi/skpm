import os
import pytest
import numpy as np
from skpm.utils import validation as v


def test_validation():
    with pytest.raises(Exception):
        v.validate_columns(input_columns=[1, 2, 3], required=[4])

    out = v.ensure_list("exception")
    assert isinstance(out, list)

    out = v.ensure_list({1, 2, 3})
    assert isinstance(out, list)


class DummyFeatures:
    FEATURES = ("beta", "alpha")
    TARGET_ONLY = ("gamma",)

    @classmethod
    def alpha(cls, X):
        return X

    @classmethod
    def beta(cls, X):
        return X

    @classmethod
    def gamma(cls, X):
        return X


def test_resolve_features_all_uses_declaration_order():
    out = v.resolve_features(DummyFeatures, "all")
    assert [name for name, _ in out] == ["beta", "alpha"]
    assert all(callable(fn) for _, fn in out)


def test_resolve_features_none_single_and_list_order():
    assert v.resolve_features(DummyFeatures, None) == []

    out = v.resolve_features(DummyFeatures, "alpha")
    assert [name for name, _ in out] == ["alpha"]

    # list keeps user order and drops duplicates
    out = v.resolve_features(DummyFeatures, ["alpha", "beta", "alpha"])
    assert [name for name, _ in out] == ["alpha", "beta"]


def test_resolve_features_unknown_raises():
    with pytest.raises(ValueError, match="Unknown feature"):
        v.resolve_features(DummyFeatures, ["alpha", "delta"])


def test_resolve_features_target_only_raises():
    with pytest.raises(ValueError, match="prediction target"):
        v.resolve_features(DummyFeatures, "gamma")


def test_resolve_features_missing_registry_raises():
    class NoRegistry:
        pass

    with pytest.raises(TypeError, match="FEATURES"):
        v.resolve_features(NoRegistry, "all")
