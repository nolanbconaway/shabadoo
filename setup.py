"""Setup."""

from pathlib import Path

from setuptools import find_packages, setup

THIS_DIRECTORY = Path(__file__).resolve().parent


INSTALL_REQUIRES = [
    # wait until numpyro makes a new release with stable deps.
    "numpyro @ git+https://github.com/pyro-ppl/numpyro.git@e7c74e76aead421748831d5d8ce3f9b6d15df33e",
    "pandas>=1.0.0,<2.0.0",  # ok w/ any 1.x pandas
]

# use readme as long description
LONG_DESCRIPTION = (THIS_DIRECTORY / "readme.md").read_text()
VERSION = (THIS_DIRECTORY / "src" / "shabadoo" / "version").read_text().strip()

setup(
    name="shabadoo",
    version=VERSION,
    description="Very easy Bayesian regression using numpyro.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Nolan Conaway",
    author_email="nolanbconaway@gmail.com",
    url="https://github.com/nolanbconaway/shabadoo",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=["bayesian", "regression", "mcmc"],
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[INSTALL_REQUIRES],
    extras_require=dict(
        test=["black", "pytest", "pytest-cov", "codecov",],
        docs=["sphinx~=2.4.4", "m2r", "numpydoc"],
    ),
    package_data={"shabadoo": ["version"]},
)
