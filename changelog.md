# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4] - 2020-03-07

- Fixes #17: Support an aggfunc option for .predict
- Fixes #15: Custom errors for better exception catching 
- Add `sampler_kwargs` to .fit() so that users can tune the sampler.

## [0.0.3] - 2020-02-23

- Fixes #14 - A bug in which the formula printer did not include plus signs.
- Fixes #11 - Allow users to set the rng seed when randomness is used (`.fit`, `.sample_posterior_predictive`). Thanks @favianrahman for the suggestion!
- Fixes #6 - Add a confidence interval option in the `.predict` method.
- Add this changelog!

## [0.0.1, 0.0.2, 0.0.2.1] - 2020-02-15

- Init releases. I made many because there were some problems with the pypi auto release.