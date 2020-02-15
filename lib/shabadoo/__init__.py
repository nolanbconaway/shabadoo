"""Models for very easy Bayesian Regression."""

from pathlib import Path

from .shabadoo import Bernoulli, Normal, Poisson

__version__ = (Path(__file__).resolve().parent / "version").read_text().strip()

__all__ = ["Normal", "Bernoulli", "Poisson", "__version__"]
