"""Test utilities unrelated to any particular model."""
import json

import jax.numpy as np
import numpy as onp
import pandas as pd
import pytest
from numpyro import distributions as dist

from shabadoo import Normal, exceptions


def test_no_feature_exception():
    """Test the exception when features are not defined."""

    class NoFeatsModel(Normal):
        dv = "y"

    with pytest.raises(exceptions.IncompleteModel) as e:
        NoFeatsModel()

    assert "NoFeatsModel" in str(e.value)
    assert "features" in str(e.value)


def test_no_dv_exception():
    """Test the exception when dv is not defined."""

    class NoDVModel(Normal):
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    with pytest.raises(exceptions.IncompleteModel) as e:
        NoDVModel()

    assert "NoDVModel" in str(e.value)
    assert "dv" in str(e.value)


def test_require_fitted_exception():
    """Test exception is raised if trying to call a required method before fitting."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    with pytest.raises(exceptions.NotFittedError) as e:
        Model().formula

    assert "formula" in str(e.value)


def test_refit_exception():
    """Test the exception raised when re-fitting."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0], y=1))

    class PreFitModel(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    model = PreFitModel().from_dict({"samples": {"x": [[1.0]], "_sigma": [[0.0]]}})

    with pytest.raises(exceptions.AlreadyFittedError) as e:
        model.fit(df)

    assert "PreFitModel" in str(e.value)


def test_invalid_sampler_exception():
    """Test the exception raised when specifying an invalid sampler."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0], y=1))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    with pytest.raises(ValueError):
        Model().fit(df, sampler="INVALID")


def test_missing_samples_exception():
    """Test the exception when creating a model from samples without a known key."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0], y=1))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    with pytest.raises(exceptions.IncompleteSamples) as e:
        Model().from_dict({"samples": {"_sigma": [[0]]}})

    assert "x" in str(e.value)

    with pytest.raises(exceptions.IncompleteSamples) as e:
        Model().from_dict({"samples": {"x": [[0]]}})

    assert "_sigma" in str(e.value)


def test_incomplete_feature_exception():
    """Test the exception when a feature is missing a key."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1))

    with pytest.raises(exceptions.IncompleteFeature) as e:
        Model()

    assert "x" in str(e.value)
    assert "prior" in str(e.value)


@pytest.mark.parametrize(
    "season", [dict(series_order=1), dict(period=1)],
)
def test_invalid_season_exception(season):
    """Test that checks for seasonality are correctly raised."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))
        seasonality = dict(my_season=season)

    with pytest.raises(exceptions.IncompleteFeature) as e:
        Model()

    assert "my_season" in str(e.value)
