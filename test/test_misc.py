"""Test utilities unrelated to any particular model."""
import numpy as onp

import shabadoo
from shabadoo.shabadoo import metrics


def test_metrics():
    """Test that metrics returns the expected values."""
    perfect = metrics(onp.array([1, 2, 3]), onp.array([1, 2, 3]))
    assert perfect == dict(r=1, rsq=1, mae=0, mape=0,)

    # dont bother with checking the nan
    terrible = metrics(onp.array([1, 2, 3]), onp.array([0, 0, 0]))
    assert terrible["mae"] == 2
    assert terrible["mape"] == 1


def test_version():
    assert isinstance(shabadoo.__version__, str)
