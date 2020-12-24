# Shabadoo: very easy Bayesian regression.

## DEV BRANCH

This is a branch i had under development for a long time.  Among other things, it:

1. implements frequency domain seasonality configurations. I still need to write tests of seasonality serialization/deserialization. Probably also sanity tests of actual fitting, predicting.
2. Starts to implement a notebook-based test. I wanted to include some sample notebooks in the docs and _also_ render them as a form of testing in a "real world" scenario. I wrote the notebooks but did not figure out how to get them in the docs.
  
Both goals are large, took a long time to get to where they are, and are both incomplete. I have largely given up on numpyro given the advances in pymc3-running-on-jax. So I do not intend on implementing these items any time soon.

## Back to the readme...

>![Imgur](https://i.imgur.com/yScWnEt.jpg)
>
> "That's the worst name I ever heard."

[![badge](https://github.com/nolanbconaway/shabadoo/workflows/Lint%20and%20Test/badge.svg)](https://github.com/nolanbconaway/shabadoo/actions?query=workflow%3A%22Lint+and+Test%22)
[![badge](https://github.com/nolanbconaway/shabadoo/workflows/Scheduled%20Testing/badge.svg)](https://github.com/nolanbconaway/shabadoo/actions?query=workflow%3A%22Scheduled+Testing%22)
[![codecov](https://codecov.io/gh/nolanbconaway/shabadoo/branch/master/graph/badge.svg?token=gIubsLSSHH)](https://codecov.io/gh/nolanbconaway/shabadoo)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/shabadoo)](https://pypi.org/project/shabadoo/)
[![PyPI](https://img.shields.io/pypi/v/shabadoo)](https://pypi.org/project/shabadoo/)

Shabadoo is the worst kind of machine learning. It automates nothing; your models will not perform well and it will be your own fault. 

> **BEWARE**. Shabadoo is in an open alpha phase. It is authored by someone who does not know how to manage open source projects. Things will change as the author identifies mistakes and corrects (?) them.

Shabadoo is for people who want to do Bayesian regression but who do not want to write probabilistic programming code. You only need to assign priors to features and pass your pandas dataframe to a `.fit()` / `.predict()` API.

Shabadoo runs on [numpyro](http://num.pyro.ai/) and is basically a wrapper around the [numpyro Bayesian regression tutorial](https://pyro.ai/numpyro/bayesian_regression.html).

- [DEV BRANCH](#dev-branch)
- [Back to the readme...](#back-to-the-readme)
- [Quickstart](#quickstart)
  - [Install](#install)
  - [Specifying a Shabadoo Bayesian model](#specifying-a-shabadoo-bayesian-model)
  - [Fitting & predicting the model](#fitting--predicting-the-model)
  - [Inspecting the model](#inspecting-the-model)
  - [Saving and recovering a saved model](#saving-and-recovering-a-saved-model)
- [Development](#development)

## Quickstart

### Install

```sh
pip install shabadoo
```

or

```sh
pip install git+https://github.com/nolanbconaway/shabadoo
```

### Specifying a Shabadoo Bayesian model

Shabadoo was designed to make it as easy as possible to test ideas about features and their priors. Models are defined using a class which contains configuration specifying how the model should behave.

You need to define a new class which inherits from one of the Shabadoo models. Currently, Normal, Poisson, and Bernoulli are implemented.

```python
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from shabadoo import Normal


# random number generator seed, to reproduce exactly.
RNG_KEY = np.array([0, 0])

class Model(Normal):
    dv = "y"
    features = dict(
        const=dict(transformer=1, prior=dist.Normal(0, 1)),
        x=dict(transformer=lambda df: df.x, prior=dist.Normal(0, 1)),
    )


df = pd.DataFrame(dict(x=[1, 2, 2, 3, 4, 5], y=[1, 2, 3, 4, 3, 5]))
```

The `dv` attribute specifies the variable you are predicting. `features` is a dictionary of dictionaries, with one item per feature. Above, two features are defined (`const` and `x`). Each feature needs a `transformer` and a `prior`. 

The transformer specifies how to obtain the feature given a source dataframe. The prior specifies your beliefs about the model's coefficient for that feature.

### Fitting & predicting the model

Shabadoo models implement the well-known `.fit` / `.predict` api pattern.

```python
model = Model().fit(df, rng_key=RNG_KEY)
# sample: 100%|██████████| 1500/1500 [00:04<00:00, 308.01it/s, 7 steps of size 4.17e-01. acc. prob=0.89]

model.predict(df)

"""
0    1.351874
1    2.219510
2    2.219510
3    3.087146
4    3.954782
5    4.822418
"""
```

#### Credible Intervals

Use `model.predict(df, ci=True)` to obtain a credible interval around the model's prediction. This interval accounts for error estimating the model's coefficients but does not account for the error around the model's point estimate (PRs welcome ya'll!).

```python
model.predict(df, ci=True)

"""
          y  ci_lower  ci_upper
0  1.351874  0.730992  1.946659
1  2.219510  1.753340  2.654678
2  2.219510  1.753340  2.654678
3  3.087146  2.663617  3.526434
4  3.954782  3.401837  4.548420
5  4.822418  4.047847  5.578753
"""
```

### Inspecting the model

Shabadoo's model classes come with a number of model inspection methods. It should be easy to understand your model's composition and with Shabadoo it is!

#### Print the model formula

The average and standard deviation of the MCMC samples are used to provide a rough sense of the coefficient in general.

```python
print(model.formula)

"""
y = (
    const * 0.48424(+-0.64618)
  + x * 0.86764(+-0.21281)
)
"""
```

#### Look at the posterior samples

Samples from fitted models can be accessed using `model.samples` (for raw device arrays) and `model.samples_df` (for a tidy DataFrame).


```python
model.samples['x']
"""
DeviceArray([[0.9443443 , 1.0215557 , 1.0401363 , 1.1768144 , 1.1752374 ,
...
"""

model.samples_df.head()
"""
                 const         x
chain sample                    
0     0       0.074572  0.944344
      1       0.214246  1.021556
      2      -0.172168  1.040136
      3       0.440978  1.176814
      4       0.454463  1.175237
"""
```

#### Measure prediction accuracy

The `Model.metrics()` method is packed with functionality. You should not have to write a lot of code to evaluate your model's prediction accuracy!

Obtaining aggregate statistics is as easy as:

```python
model.metrics(df)

{'r': 0.8646920305474705,
 'rsq': 0.7476923076923075,
 'mae': 0.5661819464378061,
 'mape': 0.21729708806356265}
```

For per-point errors, use `aggerrs=False`. A pandas dataframe will be returned that you can join on your source data using its index.

```python
model.metrics(df, aggerrs=False)

"""
   residual         pe        ape
0 -0.351874 -35.187366  35.187366
1 -0.219510 -10.975488  10.975488
2  0.780490  26.016341  26.016341
3  0.912854  22.821353  22.821353
4 -0.954782 -31.826066  31.826066
5  0.177582   3.551638   3.551638
"""
```

You can use `grouped_metrics` to understand within-group errors. Under the hood, the predicted and actual `dv` are groupby-aggregated (default sum) and metrics are computed within each group.

```python
df["group"] = [1, 1, 1, 2, 2, 2]
model.grouped_metrics(df, 'group')

{'r': 1.0,
 'rsq': 1.0,
 'mae': 0.17238043177407247,
 'mape': 0.023077819594065668}
```

```python
model.grouped_metrics(df, "group", aggerrs=False)

"""
       residual        pe       ape
group                              
1     -0.209107 -3.485113  3.485113
2     -0.135654 -1.130450  1.130450
"""
```

### Saving and recovering a saved model

Shabadoo models have `to_json` and `from_dict` methods which allow models to be saved and recovered exactly. 

```python
import json

# export to a JSON string
model_json = model.to_json()

# recover the model
model_recovered = Model.from_dict(json.loads(model_json))

# check the predictions are the same
model_recovered.predict(df).equals(model.predict(df))
True
```

## Development

To get a development installation going, set up a python 3.6 or 3.7 virtualenv however you'd like and set up an editable installation of Shabadoo like so:

```sh
$ git clone https://github.com/nolanbconaway/shabadoo.git 
$ cd shabadoo
$ pip install -e .[test]
```

You should be able to run the full test suite via:

```sh
$ tox -e py36  # or py37 if thats what you installed
```