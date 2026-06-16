"""Test points element functionality.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from emulsim.elements import ArrowsElement


@pytest.mark.parametrize("dim", [1, 2])
def test_arrows_element(dim, rng):
    """Test basic arrows elements functions."""
    n = 5  # number of droplets
    pos = rng.normal(size=(n, dim))

    element = ArrowsElement.from_position_direction(positions=pos, directions=0)
    assert element.dim == dim
    assert element.degrees_of_freedom == 2 * n * dim
    np.testing.assert_array_equal(element.positions, pos)
    np.testing.assert_array_equal(element.directions, np.zeros_like(pos))

    element.positions = 0
    np.testing.assert_array_equal(element.positions, np.zeros_like(pos))
