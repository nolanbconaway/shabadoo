"""Main module file."""
import typing
from abc import ABC, abstractmethod
from functools import wraps
from random import randint

import numpyro
import pandas as pd
from jax import numpy as np
from jax import random
from numpyro import diagnostics
from numpyro import distributions as dist
from numpyro import infer

# exception message raised when features is empty.
_no_features_message = """\
features are not defined! Define a class attribute (features) which is a dictionary of 
dictionaries like 
{
    'log_x': {
        'transformer': lambda df: np.log(df.x),    # .assign() call on DataFrame
        'prior_dist': 'Normal',                    # numpyro distribution name
        'prior_kwgs': {'loc': 0.0, 'scale': 2.0}   # passed to prior dist function
    }
}
"""


def require_fitted(f):
    """Decorate a function to require the model to be fitted for usage."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not args or not getattr(args[0], "_fitted"):
            raise TypeError(f"Unable to call {f.__name__} before fitting model.")
        return f(*args, **kwargs)

    return wrapper


class BaseModel(ABC):
    """Abstract base class from which all model families inherit."""

    features: typing.Dict[str, typing.Dict] = {}
    dv = None

    # wrap the linear combination if inputs {lc} in newlines and the link function
    # for formula printing only
    _formula_link_str = "(\n{lc}\n)"

    @staticmethod
    @abstractmethod
    def link(x):
        """Implement the link function for the model."""
        pass

    @abstractmethod
    def likelihood_func(self, yhat):
        """Return the likelihood distribution given predictions yhat."""
        pass

    def __init__(self):
        """Set some flags, do some validations, and do nothing else."""
        if not self.features:
            raise TypeError(_no_features_message)

        if not self.dv:
            raise TypeError("No dv defined!")

        for name, feature in self.features.items():
            for key in ("transformer", "prior_dist", "prior_kwgs"):
                if key not in feature:
                    raise ValueError(f"Feature {name} does not have config {key}!")

            if not hasattr(dist, feature["prior_dist"]):
                raise ValueError(
                    f"Invalid prior_dist {feature['prior_dist']} for feature {key}!"
                )

        # this will be split each time randomness is needed.
        self.rand_key = random.PRNGKey(randint(0, 10000))

        self._fitted = False

    def split_rand_key(self) -> random.PRNGKey:
        """Split the random key, assign a new key and return the subkey."""
        key, subkey = random.split(self.rand_key)
        self.rand_key = key
        return subkey

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe for model input."""
        return df.assign(
            **{
                feature_name: data["transformer"]
                for feature_name, data in cls.features.items()
            }
        )[cls.features.keys()]

    def model(self, df: pd.DataFrame):
        """Define and return samples from the model."""
        inputs = self.transform(df)
        coefs = {
            feature_name: numpyro.sample(
                feature_name, getattr(dist, data["prior_dist"])(**data["prior_kwgs"])
            )
            for feature_name, data in self.features.items()
        }

        # before the link function
        _yhat = np.sum(
            [
                inputs[feature_name].values * coefs[feature_name]
                for feature_name in self.features.keys()
            ],
            axis=0,
        )

        # apply link
        yhat = self.link(_yhat)

        if self.dv in df.columns:
            return numpyro.sample(
                self.dv, self.likelihood_func(yhat), obs=df[self.dv].values
            )
        return numpyro.sample(self.dv, self.likelihood_func(yhat))

    def fit(self, df: pd.DataFrame, sampler: str = "NUTS", **mcmc_kwargs):
        """Fit the model to a DataFrame."""
        if not hasattr(self, "_fitted"):
            raise TypeError("Cannot fit a model before init!")

        if self._fitted:
            raise TypeError("Cannot re-fit a model!")

        if sampler.upper() not in ("NUTS", "HMC"):
            raise ValueError("Invalid sampler, try NUTS or HMC.")

        sampler = getattr(infer, sampler.upper())

        # run pre-fit function
        self.pre_fit(df)

        # store fit df
        self.df = df

        # set up mcmc
        _mcmc_kwargs = dict(num_warmup=500, num_samples=1000)
        _mcmc_kwargs.update(mcmc_kwargs)
        mcmc = infer.MCMC(sampler(self.model), **_mcmc_kwargs)

        # do it
        mcmc.run(self.split_rand_key(), df=df)

        # store results
        self.samples = mcmc.get_samples()
        self._fitted = True

        return self

    def pre_fit(self, df: pd.DataFrame):
        """Additional functionality to run before fit is called.
        
        Use it to set model attributes at the class level.
        """
        pass

    @classmethod
    def pre_from_samples(self, samples: typing.Dict[str, np.ndarray]):
        """Additional functionality to run before from_samples is called.
        
        Use it to check for required additional variables, etc.
        """
        pass

    @require_fitted
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Obtain average predictions from the posterior predictive."""
        # remove dv, not needed for predict
        _df = df if self.dv not in df.columns else df.drop(columns=self.dv)

        # make array
        pred = infer.Predictive(self.model, self.samples)(
            self.split_rand_key(), df=_df
        )[self.dv].mean(axis=0)

        # return a series with a good index.
        return pd.Series(pred, index=df.index, name=self.dv)

    @property
    @require_fitted
    def samples_df(self) -> pd.DataFrame:
        """Return a DataFrame of MCMC samples."""
        return pd.DataFrame(self.samples)[self.features.keys()]

    @classmethod
    def from_samples(cls, samples: typing.Dict[str, np.ndarray]):
        """Return a pre-fitted model given its samples."""
        cls.pre_from_samples(samples)

        # check that all keys are there
        for feature_name in cls.features.keys():
            if feature_name not in samples:
                raise KeyError(f"No samples for feature {feature_name}!")

        model = cls()
        model._fitted = True

        # make jax arrays when needed
        model.samples = {k: np.device_put(v) for k, v in samples.items()}
        return model

    @property
    @require_fitted
    def formula(self):
        """Return a formula string describing the model."""
        descriptives = self.samples_df.describe()
        formula_template = f"{self.dv} = " + self._formula_link_str

        # get a string rep from each descriptive column. this will be for each feature:
        #       x * mu(+-sd)
        def get_str(x):
            return "\t" + f"""{x.name} * {x['mean']:0.5f}(+-{x['std']:0.5f})"""

        lc = "\n".join(descriptives.apply(get_str, axis=0).tolist())
        return formula_template.format(lc=lc)


class Normal(BaseModel):
    """Gaussian/normal family model for the generic regression model."""

    sigma_prior = 1.0

    @staticmethod
    def link(x):
        return x

    def likelihood_func(self, yhat):
        _sigma = numpyro.sample("_sigma", dist.Exponential(self.sigma_prior))
        return dist.Normal(yhat, _sigma)

    @classmethod
    def pre_from_samples(cls, samples: typing.Dict[str, np.ndarray]):
        if "_sigma" not in samples:
            raise KeyError("No samples for feature _sigma!")


class Poisson(BaseModel):
    """Exponential/poisson family model for rate data."""

    _formula_link_str = "exp(\n{lc}\n)"

    @staticmethod
    def link(x):
        return np.exp(x)

    def likelihood_func(self, yhat):
        return dist.Poisson(yhat)


class Bernoulli(BaseModel):
    """Logistic/bernoulli family model, for a binary response variable."""

    _formula_link_str = "logistic(\n{lc}\n)"

    @staticmethod
    def link(x):
        # likelihood accepts logits so no need to transform here
        return x

    def likelihood_func(self, logits):
        return dist.Bernoulli(logits=logits)
