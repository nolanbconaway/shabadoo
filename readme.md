# shabadoo: very easy Bayesian regression.

>![](shabadoo.jpg)
>
> "That's the worst name I ever heard."

shabadoo is the worst kind of machine learning. It automates nothing; your models will not perform well and it will be your own fault. 

shabadoo is for people who want to do Bayesian regression but who do not want to write probabilistic programming code. You only need to assign priors to features and pass your pandas dataframe to a `.fit()` / `.predict()` API.

shabadoo runs on [numpyro](http://num.pyro.ai/) and is basically a wrapper around the [numpyro Bayesian regression tutorial](https://pyro.ai/numpyro/bayesian_regression.html).




## TODO

- [ ] add prediction tests for poisson, bernoulli
- [ ] MAP estimate mode; no sampler.
- [ ] Surface HDPI of posterior predictions.
- [ ] Functionality from Prophet (Fourier seasonality, change points, holidays).
- [ ] API docs
- [ ] Add `__repr__` to base model.
- [ ] Metrics method to provide summary stats of predicted against actuals. 
    - [ ] With a `groupby` arg to allow the user to aggregate predictions within groups. I do this _constantly_.
- [ ] Validate dv range in pre_fit 
- [ ] pymc3-like visualization of sampler result
- [ ] store model fit metrics on .fit() call
