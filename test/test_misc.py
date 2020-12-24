"""Test utilities unrelated to any particular model."""
import json

import jax.numpy as np
import numpy as onp
import pytest
import pandas as pd
import shabadoo
from shabadoo.shabadoo import NumpyEncoder, metrics, columns_with_null_data


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
    """Test that the version file is included."""
    result = json.dumps(data, cls=NumpyEncoder)
    decoded = np.array(json.loads(result))
    assert onp.array_equal(onp.array(json.loads(result)), onp.array(data))


@pytest.mark.parametrize("nullval", [None, np.nan, onp.nan, pd.NA, pd.NaT])
def test_columns_with_null_data(nullval):
    """Test that the null data detector works as expected."""
    df_yes_nulls = pd.DataFrame(dict(x=[1, 2, 3, 4, nullval], y=[1, 2, 3, 2, 2]))
    df_no_nulls = pd.DataFrame(dict(x=[1, 2, 3, 4, 0], y=[1, 2, 3, 2, 2]))

    assert not columns_with_null_data(df_no_nulls)
    assert columns_with_null_data(df_yes_nulls) == ["x"]
