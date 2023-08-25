"""
Test spherical droplets elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import SphericalDroplet

from sim.elements import SphericalDropletsElement


@pytest.mark.parametrize("dim", [1, 2])
def test_spherical_droplets_element(dim):
    """test basic spherical droplets functions"""
    emulsion = [SphericalDroplet([0] * dim, 1), SphericalDroplet([1] * dim, 2)]
    element = SphericalDropletsElement.from_droplets(emulsion)
    assert element.dim == dim
    assert element.droplet_count == 2
    assert element.degrees_of_freedom == 2 * (dim + 1)

    # test basic error estimate
    for backend in ["numpy", "numba"]:
        error_estimator = element._make_error_estimator(backend=backend)
        assert error_estimator(element._data_numba, element._data_numba) == 0

    # create random element
    element = SphericalDropletsElement.from_random(
        num=3, bounds=[(0, 1)] * dim, radius=1, remove_overlapping=False
    )
    assert element.dim == dim
    assert element.droplet_count == 3
    np.testing.assert_allclose(element.data["radius"], 1)

    # create random element
    element = SphericalDropletsElement.from_random(
        num=3, bounds=[(0, 1)] * dim, radius=1, maxcount=5, remove_overlapping=False
    )
    assert element.dim == dim
    assert element.droplet_count == 3
    assert len(element.droplets) == 5
    np.testing.assert_allclose(element.data[:3]["radius"], 1)
    np.testing.assert_allclose(element.data[3:]["radius"], 0)
