"""Test utilities unrelated to any particular model."""
import json

import jax.numpy as np
import numpy as onp
import pandas as pd
import pytest

import shabadoo
from shabadoo.shabadoo import NumpyEncoder, metrics, fourier_series


def test_metrics():
    """Test that metrics returns the expected values."""
    perfect = metrics(onp.array([1, 2, 3]), onp.array([1, 2, 3]))
    assert perfect == dict(r=1, rsq=1, mae=0, mape=0,)

    # dont bother with checking the nan
    terrible = metrics(onp.array([1, 2, 3]), onp.array([0, 0, 0]))
    assert terrible["mae"] == 2
    assert terrible["mape"] == 1


def test_version():
    """Test that the version file is included."""
    assert isinstance(shabadoo.__version__, str)


@pytest.mark.parametrize(
    "data",
    [
        0.1,
        1,
        np.float32(0.1),
        np.int32(1),
        np.array([1, 2, 3]),
        onp.array([1, 2, 3]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        onp.array([[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_json_encoder(data):
    """Test that numpy data are correctly serialized and deserialized."""
    result = json.dumps(data, cls=NumpyEncoder)
    assert onp.array_equal(onp.array(json.loads(result)), onp.array(data))


@pytest.mark.parametrize("series_order", range(4, 7))
@pytest.mark.parametrize("n_days", range(10, 13))
def test_fourier_series_shape(n_days, series_order):
    """Test that the version file is included."""
    period = 7
    days = pd.date_range("2020-01-01", periods=n_days)
    result = fourier_series(days, period, series_order)
    assert result.shape[0] == n_days
    assert result.shape[1] == series_order * 2

