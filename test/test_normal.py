"""Test the normal/gaussian model.

This contains most of the tests for the base model in general, since its easy to reason
about things in a gaussian model.
"""
import json

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


def test_rand_key_splitter():
    """Test that the rand key splitter behaves as expected."""

    class Model(Normal):
        dv = "y"
        features = dict(mu=dict(transformer=1, prior=dist.Normal(0, 5)))

    model = Model()
    key = model.rand_key

    subkey = model.split_rand_key()
    assert (model.rand_key != key).all()
    assert (key != subkey).all()

    subkeys = model.split_rand_key(2)
    assert subkeys.shape[0] == 2


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


def test_require_fitted_exception():
    """Test exception is raised if trying to call a required method before fitting."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    with pytest.raises(TypeError):
        Model().formula


def test_refit_exception():
    """Test the exception raised when re-fitting."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0], y=1))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    samples = {"x": onp.array([1.0]), "_sigma": onp.array([0])}
    model = Model().from_samples(samples)

    with pytest.raises(TypeError):
        model.fit(df)


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

    with pytest.raises(KeyError):
        Model().from_samples({"_sigma": onp.array([0])})

    with pytest.raises(KeyError):
        Model().from_samples({"x": onp.array([0])})


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


def test_sample_posterior_predictive():
    """Test that sample ppd makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    samples = {
        "x": onp.array([1.0]),
        "_sigma": onp.array([0]),
    }  # use regular numpy array!!!
    model = Model().from_samples(samples)
    pred = model.sample_posterior_predictive(df)
    assert df.x.astype("float32").equals(pred.astype("float32"))

    # add hdpi
    samples = {
        "x": onp.array(onp.random.normal(size=(10,))),
        "_sigma": onp.array([1] * 10),
    }
    model = Model().from_samples(samples)
    pred = model.sample_posterior_predictive(df, hdpi=True)
    assert (pred.y < pred.hdpi_upper).all()
    assert (pred.y > pred.hdpi_lower).all()


def test_predict():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[-1, 0, 1, 2]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    samples = {"x": onp.array([1.0] * 10), "_sigma": onp.array([1] * 10)}
    model = Model().from_samples(samples)
    pred = model.predict(df)
    assert df.x.astype("float32").equals(pred.astype("float32"))


def test_formula():
    """Test that the formula is as expected."""
    samples = {
        "x1": onp.array([1.0] * 10),
        "x2": onp.array([1.0] * 10),
        "_sigma": onp.array([0] * 10),
    }

    class Model(Normal):
        dv = "y"
        features = dict(
            x1=dict(transformer=1, prior=dist.Normal(0, 1)),
            x2=dict(transformer=2, prior=dist.Normal(0, 1)),
        )

    model = Model().from_samples(samples)
    formula = model.formula
    expected = "y = (\n    x1 * 1.00000(+-0.00000)\n  + x2 * 1.00000(+-0.00000)\n)"
    print()
    print(formula)
    print(expected)
    assert formula == expected


def test_repr():
    """Test that the repr is as expected."""

    class Goober(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    repr_str = str(Goober())
    assert "Goober" in repr_str
    assert "Normal" in repr_str

    class SubModel(Goober):
        pass

    repr_str = str(SubModel())
    assert "SubModel" in repr_str
    assert "Normal" in repr_str


def test_num_samples():
    """Test that the number of samples property is as expected."""
    n = 10
    samples = {"x": onp.array([1.0] * n), "_sigma": onp.array([0] * n)}

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    assert Model.from_samples(samples).num_samples == n


def test_metrics_aggerrs(monkeypatch):
    """Mock response to test metrics while aggregating errors."""
    df = pd.DataFrame(dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4]))
    mock_predict = lambda *x, **y: df.x.rename("y")  # perfect response

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    monkeypatch.setattr(Model, "predict", mock_predict)
    model = Model().from_samples({"x": np.array([0]), "_sigma": np.array([0])})

    res = model.metrics(df)
    assert round(res["r"], 5) == 1
    assert round(res["rsq"], 5) == 1
    assert round(res["mae"], 5) == 0
    assert round(res["mape"], 5) == 0


def test_metrics_no_aggerrs(monkeypatch):
    """Mock response to test metrics while not aggregating errors."""
    df = pd.DataFrame(dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4]))
    mock_predict = lambda *x, **y: df.x.rename("y")  # perfect perf

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    monkeypatch.setattr(Model, "predict", mock_predict)
    model = Model().from_samples({"x": np.array([0]), "_sigma": np.array([0])})

    res = model.metrics(df, aggerrs=False)
    expected = pd.DataFrame(dict(residual=0, pe=0.0, ape=0.0), index=df.index)
    assert res.equals(expected)


def test_grouped_metrics_aggerrs(monkeypatch):
    """Mock response to test grouped metrics while aggregating errors."""
    df = pd.DataFrame(dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4], g=[1, 1, 2, 2]))
    mock_predict = lambda *x, **y: df.x.rename("y")

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    monkeypatch.setattr(Model, "predict", mock_predict)
    model = Model().from_samples({"x": np.array([0]), "_sigma": np.array([0])})

    res = model.grouped_metrics(df, "g")
    assert round(res["r"], 5) == 1
    assert round(res["rsq"], 5) == 1
    assert round(res["mae"], 5) == 0
    assert round(res["mape"], 5) == 0


def test_grouped_metrics_no_aggerrs(monkeypatch):
    """Mock response to test grouped metrics while not aggregating errors."""
    df = pd.DataFrame(dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4], g=[1, 1, 2, 2]))
    mock_predict = lambda *x, **y: df.x.rename("y")

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    monkeypatch.setattr(Model, "predict", mock_predict)
    model = Model().from_samples({"x": np.array([0]), "_sigma": np.array([0])})

    res = model.grouped_metrics(df, "g", aggerrs=False)
    expected = pd.DataFrame(dict(residual=0, pe=0.0, ape=0.0), index=[1, 2])
    assert res.equals(expected)


def test_samples_json():
    """Test that the samples json is as expected."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    samples = {"x": onp.array([1.0]), "_sigma": onp.array([0])}
    model = Model.from_samples(samples)
    expected = {"_sigma": [0], "x": [1.0]}

    assert json.loads(model.samples_json) == expected


def test_predict_ci():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    # ci = yhat when no variation in samples
    samples = {"x": onp.array([1.0] * 10), "_sigma": onp.array([1.0] * 10)}
    model = Model().from_samples(samples)
    pred = model.predict(df, ci=True).round(5).astype("float32")
    assert pred.y.equals(pred.ci_lower)
    assert pred.y.equals(pred.ci_upper)

    # lower < yhat < upper when some variation in samples
    samples = {"x": onp.random.normal(size=(10,)), "_sigma": onp.array([1.0] * 10)}
    model = Model().from_samples(samples)
    pred = model.predict(df, ci=True)
    assert (pred.y > pred.ci_lower).all()
    assert (pred.y < pred.ci_upper).all()


def test_posterior_predict_static_key():
    """Test that the posterior predictive is static when given a static key."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    samples = {"x": onp.array([1.0] * 10), "_sigma": onp.array([1.0] * 10)}
    model = Model().from_samples(samples)

    rng_key = np.array([0, 0])

    # are equal when rng is fixed
    df1 = model.sample_posterior_predictive(df, hdpi=True, rng_key=rng_key)
    df2 = model.sample_posterior_predictive(df, hdpi=True, rng_key=rng_key)
    assert df1.equals(df2)

    # unequal when not fixed.
    df1 = model.sample_posterior_predictive(df, hdpi=True)
    df2 = model.sample_posterior_predictive(df, hdpi=True)
    assert not df1.equals(df2)


def test_fit_static_key():
    """Test that fit is static when given a static key."""
    df = pd.DataFrame(dict(y=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    rng_key = np.array([0, 0])

    model1 = Model().fit(df, rng_key=rng_key, num_warmup=10, num_samples=20)
    model2 = Model().fit(df, rng_key=rng_key, num_warmup=10, num_samples=20)
    assert model1.samples_df.equals(model2.samples_df)


@pytest.mark.parametrize(
    "func, expected",
    [("mean", 2), ("sum", 6), ("min", 1), (onp.mean, 2), (onp.sum, 6), (onp.min, 1)],
)
def test_predict_aggfuncs(func, expected):
    """Test that fit is static when given a static key."""
    df = pd.DataFrame(dict(x=[1.0]))
    samples = {"x": onp.array([1.0, 2.0, 3.0]), "_sigma": onp.array([1.0, 1.0, 1.0])}

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    pred = Model().from_samples(samples).predict(df, aggfunc=func).iloc[0]
    assert pred == expected
