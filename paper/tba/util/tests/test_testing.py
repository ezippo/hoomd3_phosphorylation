
import pytest

from tba.util.testing import data


def test_data(data):
    with pytest.raises(RuntimeError):
        data['foo']
