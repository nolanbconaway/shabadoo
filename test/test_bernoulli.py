"""Test the poisson model."""
import numpy as onp
import pandas as pd
from numpyro import distributions as dist
from scipy.special import expit, logit

from shabadoo import Bernoulli


def logit(p):
    """Quick numpy logit function."""
    return -onp.log(1 / p - 1)


def test_single_coef_is_about_right_boolean():
    """Test that a single coef model which has a known value gets there.
    
    This is a good round trip test that fitting works.
    """
    # going to make a coef be the mean of y
    df = pd.DataFrame(dict(y=[False] * 1000 + [True] * 9000))
    expected_coef = logit(df.y.mean())

    class Model(Bernoulli):
        dv = "y"
        features = dict(mu=dict(transformer=1, prior=dist.Normal(0, 10)),)

    model = Model().fit(df, num_warmup=200, num_samples=1000, progress_bar=False)
    avg_coef = model.samples_df.describe()["mu"]["mean"]
    pct_error = abs(avg_coef - expected_coef) / expected_coef
    assert pct_error < 0.01


def test_single_coef_is_about_right_fractional():
    """Test that a single coef model which has a known value gets there.
    
    This is a good round trip test that fitting works.
    """
    # going to make a coef be the mean of y
    df = pd.DataFrame(dict(y=[0.1] * 1000 + [0.9] * 9000))
    expected_coef = logit(df.y.mean())

    class Model(Bernoulli):
        dv = "y"
        features = dict(mu=dict(transformer=1, prior=dist.Normal(0, 10)),)

    model = Model().fit(df, num_warmup=200, num_samples=1000, progress_bar=False)
    avg_coef = model.samples_df.describe()["mu"]["mean"]
    pct_error = abs(avg_coef - expected_coef) / expected_coef
    assert pct_error < 0.01


def test_formula():
    """Test that the formula is as expected."""
    config = {"samples": {"x1": onp.ones((2, 10)), "x2": onp.ones((2, 10))}}

    class Model(Bernoulli):
        dv = "y"
        features = dict(
            x1=dict(transformer=1, prior=dist.Normal(0, 1)),
            x2=dict(transformer=2, prior=dist.Normal(0, 1)),
        )

    model = Model.from_dict(config)
    formula = model.formula
    expected = (
        "y = logistic(\n    x1 * 1.00000(+-0.00000)\n  + x2 * 1.00000(+-0.00000)\n)"
    )
    assert formula == expected


def test_sample_posterior_predictive():
    """Test that sample ppd makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[-1, 0, 1, 2]))

    class Model(Bernoulli):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    config = {"samples": {"x": onp.ones((10, 100000))}}
    model = Model.from_dict(config)
    pred = model.sample_posterior_predictive(df)
    logit_pred = logit(pred).round(2)
    assert df.x.astype("float32").equals(logit_pred.astype("float32"))


def test_predict():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[-1, 0, 1, 2]))

    class Model(Bernoulli):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    config = {"samples": {"x": onp.ones((2, 10))}}
    model = Model.from_dict(config)
    pred = model.predict(df)
    logit_pred = logit(pred).round(2)
    assert df.x.astype("float32").equals(logit_pred.astype("float32"))


def test_predict_ci():
    """Test that predict makes sense when init from samples."""
    df = pd.DataFrame(dict(x=[1.0, 2.0, 3.0, 4.0]))

    class Model(Bernoulli):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 1)))

    # ci = yhat when no variation in samples
    config = {"samples": {"x": onp.ones((2, 10))}}
    model = Model.from_dict(config)
    pred = model.predict(df, ci=True).round(5).astype("float32")
    assert pred.y.equals(pred.ci_lower)
    assert pred.y.equals(pred.ci_upper)

    # lower < yhat < upper when some variation in samples
    config = {"samples": {"x": onp.random.normal(size=(2, 100)) * 0.1}}
    model = Model.from_dict(config)
    pred = model.predict(df, ci=True)
    assert (pred.y > pred.ci_lower).all()
    assert (pred.y < pred.ci_upper).all()
