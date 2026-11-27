"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import special

from emulsim.utils.bessel import i0, i1, k0, k1


@pytest.mark.parametrize(
    ("func1", "func2"),
    [(i0, special.i0), (i1, special.i1), (k0, special.k0), (k1, special.k1)],
)
def test_bessel(func1, func2):
    """Test implementation of bessel function."""
    xs = np.geomspace(1e-5, 1e5, 32)
    ys1 = [func1(x) for x in xs]
    ys2 = func2(xs)
    np.testing.assert_allclose(ys1, ys2, rtol=1e-6)
