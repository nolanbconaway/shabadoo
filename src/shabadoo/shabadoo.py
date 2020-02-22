"""Shabadoo models for very easy regression."""
import inspect
import json
import typing
from abc import ABC, abstractmethod
from functools import wraps
from random import randint

import numpy as onp
import numpyro
import pandas as pd
from jax import numpy as np
from jax import random
from jax.scipy.special import expit
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
        'prior': dist.Normal(0.0, 1.0),            # numpyro distribution
    }
}
"""


def require_fitted(f):
    """Decorate a function to require the model to be fitted for usage."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not args or not getattr(args[0], "fitted"):
            raise TypeError(f"Unable to call {f.__name__} before fitting model.")
        return f(*args, **kwargs)

    return wrapper


def metrics(y: pd.Series, yhat: pd.Series) -> typing.Dict[str, float]:
    """Return general fit metrics of one series against another."""
    res = dict()
    if y.shape[0] >= 2:
        res["r"] = onp.corrcoef(y, yhat)[0][1]
        res["rsq"] = res["r"] ** 2
    res["mae"] = onp.mean(onp.abs(y - yhat))
    res["mape"] = onp.mean(onp.abs(y - yhat) / y)
    return res


class BaseModel(ABC):
    """Abstract base class from which all model families inherit."""

    features: typing.Dict[str, typing.Dict[str, typing.Any]] = None
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
        """Initialize the model opbject. Set some flags and runs some validations."""
        if not self.features:
            raise TypeError(_no_features_message)

        if not self.dv:
            raise TypeError("No dv defined!")

        for name, feature in self.features.items():
            for key in ("transformer", "prior"):
                if key not in feature:
                    raise ValueError(f"Feature {name} does not have config {key}!")

        # this will be split each time randomness is needed.
        self.rand_key = random.PRNGKey(randint(0, 10000))

        self.fitted = False

    def __repr__(self):
        """Print out the model name and the model family."""
        my_name = self.__class__.__name__
        mro = inspect.getmro(self.__class__)
        model_type = next(filter(lambda x: x[1] == BaseModel, zip(mro, mro[1:])), None)

        if model_type is None:
            return f"<Shabadoo Model: {my_name}>"

        model_type = model_type[0].__name__
        return f"<Shabadoo {model_type} Model: {my_name}>"

    def split_rand_key(self, n: int = 1) -> random.PRNGKey:
        """Split the random key, assign a new key and return the subkeys.
        
        Parameters
        ----------
        n : int
            Number of subkeys to generate. Default 1.

        Returns
        -------
        random.PRNGKey
            An array of PRNG keys or just a single key (if n=1).

        """
        keys = random.split(self.rand_key, n + 1)
        self.rand_key = keys[0]
        if n == 1:
            return keys[1]
        else:
            return keys[1:]

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe for model input.
        
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe to transform.

        Returns
        -------
        pd.DataFrame
            Dataframe containing transformed inputs.

        """
        return df.assign(
            **{
                feature_name: data["transformer"]
                for feature_name, data in cls.features.items()
            }
        )[cls.features.keys()]

    def model(self, df: pd.DataFrame):
        """Define and return samples from the model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data for the model.

        """
        inputs = self.transform(df)
        coefs = {
            feature_name: numpyro.sample(feature_name, data["prior"])
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

        # we should never see an unfitted model being run without the dv
        # as the dv should always be present for fitting
        if not self.fitted and self.dv not in df.columns:
            raise ValueError("Cannot run an unfitted model without the dv present!")

        # if model is not fitted, this MUST be the call to .fit(), so add the obs.
        if not self.fitted:
            return numpyro.sample(
                self.dv, self.likelihood_func(yhat), obs=df[self.dv].values
            )

        # otherwise, this is a post-fit predict call or something, do not include an obs.
        return numpyro.sample(self.dv, self.likelihood_func(yhat))

    def fit(self, df: pd.DataFrame, sampler: str = "NUTS", **mcmc_kwargs):
        """Fit the model to a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe.
        sampler : str
            Numpyro sampler name. Default NUTS
        **mcmc_kwargs :
            Passed to numpyro.infer.MCMC

        Returns
        -------
        Model
            The fitted model.

        """
        if self.fitted:
            raise TypeError("Cannot re-fit a model!")

        if sampler.upper() not in ("NUTS", "HMC"):
            raise ValueError("Invalid sampler, try NUTS or HMC.")

        sampler = getattr(infer, sampler.upper())

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
        self.fitted = True

        return self

    @classmethod
    def pre_from_samples(self, samples: typing.Dict[str, np.ndarray]):
        """Additional functionality to run before from_samples is called.
        
        Use it to check for required additional variables, etc.
        """
        pass

    def predict(
        self, df: pd.DataFrame, ci: bool = False, ci_interval: float = 0.9
    ) -> typing.Union[pd.Series, pd.DataFrame]:
        """Return the average posterior prediction across all samples.
        
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe.
        ci : float
            Option to include a confidence interval around the predictions. Returns a 
            dataframe if true, a series if false. Default False.
        ci_interval : float
            Confidence interval width. Default 0.9.

        Returns
        -------
        pd.Series or pd.DataFrame
            Forecasts. Will be a series with the name of the dv if no ci. Will be a
            dataframe if ci is included.

        """
        # matmul inputs * coefs, then send through link
        predictions = self.link(
            self.transform(df).values @ self.samples_df.transpose().values
        )

        if not ci:
            return pd.Series(predictions.mean(axis=1), index=df.index, name=self.dv)

        quantiles = onp.quantile(predictions, [1 - ci_interval, ci_interval], axis=1)
        return pd.DataFrame(
            {
                self.dv: predictions.mean(axis=1),
                "ci_lower": quantiles[0, :],
                "ci_upper": quantiles[1, :],
            },
            index=df.index,
        )

    @require_fitted
    def sample_posterior_predictive(
        self, df: pd.DataFrame, hdpi: bool = False, hdpi_interval: float = 0.9
    ) -> typing.Union[pd.Series, pd.DataFrame]:
        """Obtain samples from the posterior predictive.
        
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe.
        hdpi : bool
            Option to include lower/upper bound of the higher posterior density 
            interval. Returns a dataframe if true, a series if false. Default False.
        hdpi_interval : float
            HDPI width. Default 0.9.

        Returns
        -------
        pd.Series or pd.DataFrame
            Forecasts. Will be a series with the name of the dv if no HDPI. Will be a
            dataframe if HDPI is included.

        """
        predictions = infer.Predictive(self.model, self.samples)(
            self.split_rand_key(), df=df
        )[self.dv]

        if not hdpi:
            return pd.Series(predictions.mean(axis=0), index=df.index, name=self.dv)

        hdpi = diagnostics.hpdi(predictions, hdpi_interval)

        return pd.DataFrame(
            {
                self.dv: predictions.mean(axis=0),
                "hdpi_lower": hdpi[0, :],
                "hdpi_upper": hdpi[1, :],
            },
            index=df.index,
        )

    @property
    @require_fitted
    def samples_df(self) -> pd.DataFrame:
        """Return a DataFrame of the model's MCMC samples."""
        return pd.DataFrame(self.samples)[self.features.keys()]

    @property
    @require_fitted
    def samples_json(self) -> str:
        """Return a JSON payload of the model's MCMC samples."""
        return json.dumps({k: list(map(float, v)) for k, v in self.samples.items()})

    @classmethod
    def from_samples(cls, samples: typing.Dict[str, np.ndarray]):
        """Return a pre-fitted model given its samples.
        
        Parameters
        ----------
        samples : dict mapping string feature name to -> numpy array samples.
            MCMC samples.

        Returns
        -------
        Model
            A ready-to-use model.
        """
        cls.pre_from_samples(samples)

        # check that all keys are there
        for feature_name in cls.features.keys():
            if feature_name not in samples:
                raise KeyError(f"No samples for feature {feature_name}!")

        model = cls()
        model.fitted = True

        # make jax arrays when needed
        model.samples = {k: np.device_put(v) for k, v in samples.items()}
        return model

    @property
    @require_fitted
    def num_samples(self) -> int:
        """Return the number of samples available."""
        return list(self.samples.values())[0].shape[0]

    @require_fitted
    def metrics(
        self, df: pd.DataFrame, aggerrs: bool = True
    ) -> typing.Union[pd.DataFrame, typing.Dict[str, float]]:
        """Get prediction accuracy metrics of the model against data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data for the model.
        aggerrs : bool
            Option to aggregate errors across observations (default True). If true, 
            a dictionary of summary statistics are returned. If False, pointwise errors
            are returned as a DataFrame.
        
        Returns
        -------
        dict or pd.DataFrame
            If aggerrs, a dictionary of summary statistics are returned. If False, 
            pointwise errors are returned as a DataFrame.
        """
        y = df[self.dv]
        yhat = self.predict(df)
        if aggerrs:
            return metrics(y, yhat)

        return pd.DataFrame(dict(residual=y - yhat), index=df.index).assign(
            pe=lambda x: x.residual / y * 100, ape=lambda x: onp.abs(x["pe"]),
        )

    @require_fitted
    def grouped_metrics(
        self,
        df: pd.DataFrame,
        groupby: typing.Union[str, typing.List[str]],
        aggfunc: typing.Callable = onp.sum,
        aggerrs: bool = True,
    ) -> typing.Union[pd.DataFrame, typing.Dict[str, float]]:
        """Return grouped accuracy metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data for the model.
        groupby : str or list of str
            Groupby clause for pandas.
        aggfunc : callable
            How to aggregate actuals and predictions wihtin a group. Default sum.
        aggerrs : bool
            Option to aggregate errors across groups (default True). If true, 
            a dictionary of summary statistics are returned. If False, groupwise errors
            are returned as a DataFrame.
        
        Returns
        -------
        dict or pd.DataFrame
            If aggerrs, a dictionary of summary statistics are returned. If False, 
            groupwise errors are returned as a DataFrame.
        """
        # aggregate per group
        res = (
            df.assign(_yhat=self.predict)
            .groupby(groupby)
            .agg(**{"y": (self.dv, aggfunc), "yhat": ("_yhat", aggfunc),})
            .assign(
                residual=lambda x: x.yhat - x.y,
                pe=lambda x: x.residual / x.y * 100,
                ape=lambda x: onp.abs(x["pe"]),
            )
        )

        if not aggerrs:
            return res[["residual", "pe", "ape"]]
        return metrics(res.y, res.yhat)

    @property
    @require_fitted
    def formula(self):
        """Return a formula string describing the model."""
        descriptives = self.samples_df.describe()
        formula_template = f"{self.dv} = " + self._formula_link_str
        print(descriptives)
        # get a string rep from each descriptive column. this will be for each feature:
        #       x * mu(+-sd)
        def get_str(x):
            mu = x["mean"]
            sd = x["std"]
            if x.name == descriptives.columns.values[0]:
                prefix = "    "
            else:
                prefix = "  + "
            return prefix + f"""{x.name} * {mu:0.5f}(+-{sd:0.5f})"""

        lc = "\n".join(descriptives.apply(get_str, axis=0).tolist())
        return formula_template.format(lc=lc)


class Normal(BaseModel):
    """Gaussian/normal family model for the generic regression model."""

    sigma_prior = 1.0

    @staticmethod
    def link(x):
        """Linear link function."""
        return x

    def likelihood_func(self, yhat):
        """Return a normal likelihood with fitted sigma."""
        _sigma = numpyro.sample("_sigma", dist.Exponential(self.sigma_prior))
        return dist.Normal(yhat, _sigma)

    @classmethod
    def pre_from_samples(cls, samples: typing.Dict[str, np.ndarray]):
        """Check for sigma before init from samples."""
        if "_sigma" not in samples:
            raise KeyError("No samples for feature _sigma!")


class Poisson(BaseModel):
    """Exponential/poisson family model for rate data."""

    _formula_link_str = "exp(\n{lc}\n)"

    @staticmethod
    def link(x):
        """Exponential link function."""
        return np.exp(x)

    def likelihood_func(self, yhat):
        """Return a poisson likelihood."""
        return dist.Poisson(yhat)


class Bernoulli(BaseModel):
    """Logistic/bernoulli family model, for a binary response variable."""

    _formula_link_str = "logistic(\n{lc}\n)"

    @staticmethod
    def link(x):
        """Logistic link function."""
        return expit(x)

    def likelihood_func(self, probs):
        """Return a Bernoulli likelihood."""
        return dist.Bernoulli(probs=probs)
