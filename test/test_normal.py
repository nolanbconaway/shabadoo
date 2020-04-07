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

import shabadoo.exceptions as exceptions
from shabadoo import Normal

try:
    numpyro.set_host_device_count(2)
except Exception:
    print("Failed to set device count")


def test_single_coef_is_about_right():
    """Test that a single coef model which has a known value gets there.
    
    This is a good round trip test that fitting works.
    """
    # going to make a coef be the mean of y
    df = pd.DataFrame(dict(y=[1, 2, 2, 2, 2, 3] * 100))

    class Model(Normal):
        dv = "y"
        features = dict(mu=dict(transformer=1, prior=dist.Normal(0, 5)))

    model = Model().fit(
        df, num_chains=2, num_warmup=100, num_samples=200, progress_bar=False
    )
    assert model.num_samples == 2 * 200
    assert model.num_chains == 2
    avg_x_coef = model.samples_df.describe()["mu"]["mean"]

    assert abs(avg_x_coef - 2) < 0.1


@pytest.mark.parametrize("samples_per_chain", [10, 20, 100, 1000])
@pytest.mark.parametrize("chains", range(1, 3))
def test_samples_df(samples_per_chain, chains):
    """Test that the samples df is as expected."""

    class Model(Normal):
        dv = "y"
        features = dict(
            x1=dict(transformer=1, prior=dist.Normal(0, 1)),
            x2=dict(transformer=2, prior=dist.Normal(0, 1)),
        )

    config = {
        "samples": {
            "x1": onp.random.normal(size=(chains, samples_per_chain)),
            "x2": onp.random.normal(size=(chains, samples_per_chain)),
            "_sigma": onp.ones((chains, samples_per_chain)),
        }
    }

    model = Model.from_dict(config)
    df = model.samples_df.reset_index()
    assert df["chain"].max() == (chains - 1)
    assert df["sample"].max() == (samples_per_chain - 1)
    assert df.shape[0] == (samples_per_chain * chains)
    assert df.shape[1] == 4  # 2 features, plus two indecies
    for x in model.features.keys():
        assert np.array_equal(df[x].values, config["samples"][x].flatten())
        config_first_chain = config["samples"][x][0, :]
        df_first_chain = df.sort_values("sample").loc[lambda x: x.chain == 0, x].values
        assert np.array_equal(df_first_chain, config_first_chain)


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

    config = {"samples": {"x": [[1.0]], "_sigma": [[0.0]]}}
    model = Model.from_dict(config)
    pred = model.sample_posterior_predictive(df)
    assert df.x.astype("float32").equals(pred.astype("float32"))

    # add hdpi
    config = {
        "samples": {"x": onp.random.normal(size=(2, 10)), "_sigma": onp.ones((2, 10))}
    }
    model = Model.from_dict(config)
    pred = model.sample_posterior_predictive(df, hdpi=True)
    assert (pred.y < pred.hdpi_upper).all()
    assert (pred.y > pred.hdpi_lower).all()


def test_predict():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[-1, 0, 1, 2]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    config = {"samples": {"x": np.ones((2, 10)), "_sigma": np.ones((2, 10))}}
    model = Model.from_dict(config)
    pred = model.predict(df)
    assert df.x.astype("float32").equals(pred.astype("float32"))


def test_formula():
    """Test that the formula is as expected."""
    config = {
        "samples": {
            "x1": onp.ones((2, 10)),
            "x2": onp.ones((2, 10)),
            "_sigma": onp.zeros((2, 10)),
        }
    }

    class Model(Normal):
        dv = "y"
        features = dict(
            x1=dict(transformer=1, prior=dist.Normal(0, 1)),
            x2=dict(transformer=2, prior=dist.Normal(0, 1)),
        )

    model = Model.from_dict(config)
    formula = model.formula
    expected = "y = (\n    x1 * 1.00000(+-0.00000)\n  + x2 * 1.00000(+-0.00000)\n)"
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


@pytest.mark.parametrize("samples_per_chain", [10, 20, 100, 1000])
@pytest.mark.parametrize("chains", range(1, 3))
def test_num_samples_num_chains(samples_per_chain, chains):
    """Test that the number of samples and chains property are as expected."""
    config = {
        "samples": {
            "x": onp.ones((chains, samples_per_chain)),
            "_sigma": onp.ones((chains, samples_per_chain)),
        }
    }

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    assert Model.from_dict(config).num_chains == chains
    assert Model.from_dict(config).num_samples == samples_per_chain * chains


def test_metrics_aggerrs(monkeypatch):
    """Mock response to test metrics while aggregating errors."""
    df = pd.DataFrame(dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4]))
    mock_predict = lambda *x, **y: df.x.rename("y")  # perfect response

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    monkeypatch.setattr(Model, "predict", mock_predict)
    model = Model().from_dict({"samples": {"x": [[0]], "_sigma": [[0.0]]}})

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
    model = Model().from_dict({"samples": {"x": [[0]], "_sigma": [[0.0]]}})

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
    model = Model().from_dict({"samples": {"x": [[0]], "_sigma": [[0.0]]}})

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
    model = Model().from_dict({"samples": {"x": [[0]], "_sigma": [[0.0]]}})

    res = model.grouped_metrics(df, "g", aggerrs=False)
    expected = pd.DataFrame(dict(residual=0, pe=0.0, ape=0.0), index=[1, 2])
    assert res.equals(expected)


def test_to_json():
    """Test that the to_json payload is as expected."""

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    config = {"samples": {"x": [[1.0]], "_sigma": [[0.0]]}}
    expected = {"samples": {"_sigma": [[0.0]], "x": [[1.0]]}}
    model = Model.from_dict(config)
    assert json.loads(model.to_json()) == expected


def test_predict_ci():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    # ci = yhat when no variation in samples
    config = {"samples": {"x": onp.ones((2, 10)), "_sigma": onp.ones((2, 10))}}
    model = Model.from_dict(config)
    pred = model.predict(df, ci=True).round(5).astype("float32")
    assert pred.y.equals(pred.ci_lower)
    assert pred.y.equals(pred.ci_upper)

    # lower < yhat < upper when some variation in samples
    config = {
        "samples": {"x": onp.random.normal(size=(2, 10)), "_sigma": onp.ones((2, 10))}
    }
    model = Model.from_dict(config)
    pred = model.predict(df, ci=True)
    assert (pred.y > pred.ci_lower).all()
    assert (pred.y < pred.ci_upper).all()


def test_posterior_predict_static_key():
    """Test that the posterior predictive is static when given a static key."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=1, prior=dist.Normal(0, 1)))

    config = {"samples": {"x": onp.ones((2, 10)), "_sigma": onp.ones((2, 10))}}
    model = Model.from_dict(config)

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
    config = {
        "samples": {
            "x": onp.array([[1.0, 2.0, 3.0]]),
            "_sigma": onp.array([[1.0, 1.0, 1.0]]),
        }
    }

    class Model(Normal):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    pred = Model.from_dict(config).predict(df, aggfunc=func).iloc[0]
    assert pred == expected
