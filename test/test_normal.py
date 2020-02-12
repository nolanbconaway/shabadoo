"""Test the normal/gaussian model.

This contains most of the tests for the base model in general, since its easy to reason
about things in a gaussian model.
"""

import jax.numpy as np
import numpy as onp
import numpyro
import pandas as pd
import pytest
from numpyro import distributions as dist

from shabadoo import Normal


def test_single_coef_is_about_right():
    """Test that a single coef model which has a known value gets there.
    
    This is a good round trip test that fitting works.
    """
    # going to make a coef be the mean of y
    df = pd.DataFrame(dict(y=[1, 2, 2, 2, 3]))

    class Model(Normal):
        dv = "y"
        features = dict(mu=dict(transformer=1, prior=dist.Normal(0, 5)))

    model = Model().fit(df, num_warmup=200, num_samples=500, progress_bar=False)
    avg_x_coef = model.samples_df.describe()["mu"]["mean"]

    assert abs(avg_x_coef - 2) < 0.1


def test_prediction_with_assumed_samples(monkeypatch):
    """Test that the posterior predictive works.
    
    monkeypatch fitting for speed.
    """
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    # model will not be fit so dont need to set much.
    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    # monkeypatch fitted samples
    fake_samples = {"x": np.array([1.0]), "_sigma": np.array([0])}
    monkeypatch.setattr(numpyro.infer.MCMC, "run", lambda *x, **y: None)
    monkeypatch.setattr(numpyro.infer.MCMC, "get_samples", lambda *x, **y: fake_samples)

    pred = Model().fit(df).predict(df)
    assert df.x.astype("float32").equals(pred.astype("float32"))


def test_no_feature_exception():
    """Test the exception when features are not defined."""

    class Model(Normal):
        dv = "y"

    with pytest.raises(TypeError):
        Model()


def test_no_dv_exception():
    """Test the exception when dv is not defined."""

    class Model(Normal):
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    with pytest.raises(TypeError):
        Model()


def test_incomplete_feature_exception():
    """Test the exception when a feature is missing a key."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1))

    with pytest.raises(ValueError):
        Model()


def test_transform():
    """Test the transform method returns the expected dataframe."""

    class Model(Normal):
        dv = "y"
        features = dict(
            one=dict(transformer=1, prior=dist.Normal(0, 1)),
            x_sq=dict(transformer=lambda x: x.x ** 2, prior=dist.Normal(0, 1)),
        )

    df = pd.DataFrame(dict(x=[1, 2, 3], y=1))
    expected = pd.DataFrame(dict(one=1, x_sq=[1, 4, 9]))
    assert Model.transform(df).equals(expected)


def test_from_samples_predict():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    samples = {
        "x": onp.array([1.0]),
        "_sigma": onp.array([0]),
    }  # use regular numpy array!!!
    model = Model().from_samples(samples)
    pred = model.predict(df)
    assert df.x.astype("float32").equals(pred.astype("float32"))


def test_formula():
    """Test that the formula is as expected."""
    samples = {"x": onp.array([1.0] * 10), "_sigma": onp.array([0] * 10)}

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    model = Model().from_samples(samples)
    formula = model.formula
    expected = "y = (\n\tx * 1.00000(+-0.00000)\n)"
    assert formula == expected
