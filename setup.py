"""Setup."""

from setuptools import find_packages, setup

INSTALL_REQUIRES = ["numpyro~=0.2.4", "pandas~=1.0.1"]

setup(
    name="shabadoo",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[INSTALL_REQUIRES],
    extras_require=dict(test=["black", "pytest"]),
)
