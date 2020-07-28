# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2020-07-27

0.1.2 came a day too soon! Numpyro created a new release (0.3.0) with stable dependencies and this release migrates to that version.

## [0.1.2] - 2020-07-26

Another bugfix release, no changes to the API. Numpyro's dependencies have not stablized yet so I am pinning to a specific commit (the latest one at the time I pinned it).

- Continued Fixes #12: reduce hackyness of the numpyro/jax installation step in CI.

## [0.1.1] - 2020-05-16

A bugfix release, no changes to the API. Users with older versions of jax/jaxlib may 
need to upgrade to `jax~=0.1.67` / `jaxlib~=0.1.47` as some parts of the jax API have 
changed. Shabadoo has changed how it imports functions from jax and will not import 
correctly with older jax versions.

- Fixes #24: set `rng_key` for tests that involve randomness.
- Fixes #12: reduce hackyness of the numpyro/jax installation step in CI.

## [0.1.0] - 2020-04-06

- Fixes #20 - Preserve chain grouping on .fit. Most other changes inservice of this.
- Alter json export and import to support 2d samples.
- Change all tests to use 2d samples.
- Nest samples within a config dict to allow new fields for import/export.
- Use custom JSONEncoder.
- Nix pre_from_samples in favor of class attributes.
- Rename from_samples to from_dict.
- Make distinct config preprocessor.

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
