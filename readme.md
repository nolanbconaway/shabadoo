# Shabadoo: very easy Bayesian regression.

>![Imgur](https://i.imgur.com/yScWnEt.jpg)
>
> "That's the worst name I ever heard."

[![badge](https://github.com/nolanbconaway/shabadoo/workflows/Lint%20and%20Test/badge.svg)](https://github.com/nolanbconaway/shabadoo/actions?query=workflow%3A%22Lint+and+Test%22)
[![codecov](https://codecov.io/gh/nolanbconaway/shabadoo/branch/master/graph/badge.svg?token=gIubsLSSHH)](https://codecov.io/gh/nolanbconaway/shabadoo)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/shabadoo)](https://pypi.org/project/shabadoo/)
[![PyPI](https://img.shields.io/pypi/v/shabadoo)](https://pypi.org/project/shabadoo/)

Shabadoo is the worst kind of machine learning. It automates nothing; your models will not perform well and it will be your own fault. 

Shabadoo is for people who want to do Bayesian regression but who do not want to write probabilistic programming code. You only need to assign priors to features and pass your pandas dataframe to a `.fit()` / `.predict()` API.

Shabadoo runs on [numpyro](http://num.pyro.ai/) and is basically a wrapper around the [numpyro Bayesian regression tutorial](https://pyro.ai/numpyro/bayesian_regression.html).

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
import pandas as pd
from numpyro import distributions as dist
from shabadoo import Normal

# fake data
df = pd.DataFrame(dict(x=[1, 2, 2, 3, 4, 5], y=[1, 2, 3, 4, 3, 5]))

class Model(Normal):
    dv = "y"
    features = dict(
        const=dict(transformer=1, prior=dist.Normal(0, 1)),
        x=dict(transformer=lambda df: df.x, prior=dist.Normal(0, 1)),
    )

```

The `dv` attribute specifies the variable you are predicting. `features` is a dictionary of dictionaries, with one item per feature. Above, two features are defined (`const` and `x`). Each feature needs a `transformer` and a `prior`. 

The transformer specifies how to obtain the feature given a source dataframe. The prior specifies your beliefs about the model's coefficient for that feature.

### Fitting & predicting the model

Shabadoo models implement the well-known `.fit` / `.predict` api pattern.

```python
model = Model().fit(df)
# sample: 100%|██████████| 1500/1500 [00:05<00:00, 282.76it/s, 7 steps of size 4.17e-01. acc. prob=0.88]

model.predict(df)

"""
0    1.309280
1    2.176555
2    2.176555
3    3.043831
4    3.911106
5    4.778381
"""
```

### Inspecting the model

Shabadoo's model classes come with a number of model inspection methods. It should be easy to understsand your model's composition and with Shabadoo it is!

#### Print the model formula

The average and standard deviation of the MCMC samples are used to provide a rough sense of the coefficient in general.

```python
print(model.formula)

"""
y = (
	const * 0.44200(+-0.63186)
	x * 0.86728(+-0.22604)
)
"""
```

#### Measure prediction accuracy.

The `Model.metrics()` method is packed with functionality. You should not have to write a lot of code to evaluate your model's prediction accuracy!

Obtaining aggregate statistics is as easy as:

```python
model.metrics(df)

{'r': 0.8646920305474705,
 'rsq': 0.7476923076923075,
 'mae': 0.5663623639121652,
 'mape': 0.20985123644135573}
```

For per-point errors, use `aggerrs=False`. A pandas dataframe will be returned that you can join on your source data using its index.

```python
model.metrics(df, aggerrs=False)

"""
   residual         pe        ape
0 -0.309280 -30.928012  30.928012
1 -0.176555  -8.827769   8.827769
2  0.823445  27.448154  27.448154
3  0.956169  23.904233  23.904233
4 -0.911106 -30.370198  30.370198
5  0.221619   4.432376   4.432376
"""
```

You can use `grouped_metrics` to understand within-group errors. Under the hood, the predicted and actual `dv` are groupby-aggregated (default sum) and metrics are computed within each group.

```python
df["group"] = [1, 1, 1, 2, 2, 2]
model.grouped_metrics(df, 'group')

{'r': 1.0, 'rsq': 1.0, 'mae': 0.30214565559127315, 'mape': 0.03924585080786096}
```

```python
model.grouped_metrics(df, "group", aggerrs=False)

"""
       residual        pe       ape
group                              
1     -0.337609 -5.626818  5.626818
2     -0.266682 -2.222352  2.222352
"""
```

### Saving and recovering a saved model

Shabadoo models have a `from_samples` method which allows a model to be save and recovered exactly. 

Samples from fitted models can be accessed using `model.samples` and `model.samples_df`.


```python
model.samples['x']
"""
DeviceArray([0.65721655, 0.7644873 , 0.8724553 , 0.6285299 , 0.681262  ,
...
"""

model.samples_df.head()
"""
      const         x
0  0.689248  0.657217
1  0.524834  0.764487
2  1.093962  0.872455
3  1.253354  0.628530
4  1.021025  0.681262
"""
```

Use the samples to recover your model:

```python
model_recovered = Model.from_samples(model.samples)

model_recovered.predict(df).equals(model.predict(df))
True
```

Model samples can be saved as JSON using `model.samples_json`:

```python
import json

with open('model.json', 'w') as f:
    f.write(model.samples_json)

with open('model.json', 'r') as f:
    model_recovered = Model.from_samples(json.load(f))
```