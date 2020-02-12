"""Test the poisson model."""

import numpy as onp
import pandas as pd
from numpyro import distributions as dist

from shabadoo import Poisson


def test_single_coef_is_about_right():
    """Test that a single coef model which has a known value gets there.
    
    This is a good round trip test that fitting works.
    """
    # going to make a coef be the mean of y
    df = pd.DataFrame(dict(y=[10, 11, 11, 11, 12]))
    expected_coef = onp.log(df.y).mean()

    class Model(Poisson):
        dv = "y"
        features = dict(mu=dict(transformer=1, prior=dist.Normal(0, 5)),)

    model = Model().fit(df, num_warmup=200, num_samples=500, progress_bar=False)
    avg_coef = model.samples_df.describe()["mu"]["mean"]
    pct_error = abs(avg_coef - expected_coef) / expected_coef
    assert pct_error < 0.01


def test_formula():
    """Test that the formula is as expected."""
    samples = {"x": onp.array([1.0] * 10)}

    class Model(Poisson):
        dv = "y"
        features = dict(x=dict(transformer=lambda x: x.x, prior=dist.Normal(0, 10)))

    model = Model().from_samples(samples)
    formula = model.formula
    expected = "y = exp(\n\tx * 1.00000(+-0.00000)\n)"
    assert formula == expected
