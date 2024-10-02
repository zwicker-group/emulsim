"""Test spherical droplets elements functionality.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import Emulsion, SphericalDroplet

from sim.elements import SphericalDropletsElement


@pytest.mark.parametrize("dim", [1, 2])
def test_spherical_droplets_element(dim, rng):
    """Test basic spherical droplets functions."""
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
        num=3, bounds=[(0, 1)] * dim, radius=1, remove_overlapping=False, rng=rng
    )
    assert element.dim == dim
    assert element.droplet_count == 3
    np.testing.assert_allclose(element.data["radius"], 1)

    # create random element
    element = SphericalDropletsElement.from_random(
        num=3,
        bounds=[(0, 1)] * dim,
        radius=1,
        maxcount=5,
        remove_overlapping=False,
        rng=rng,
    )
    assert element.dim == dim
    assert element.droplet_count == 3
    assert len(element.droplets) == 5
    np.testing.assert_allclose(element.data[:3]["radius"], 1)
    np.testing.assert_allclose(element.data[3:]["radius"], 0)


@pytest.mark.parametrize("dim", [1, 2])
def test_spherical_droplets_element_empty(dim):
    """Test empty spherical droplets elements."""
    d = SphericalDroplet([0] * dim, 1)

    def check(el, size):
        assert el.droplet_count == 0
        assert el.data.size == size
        assert el.dim == dim

    # really empty
    el = SphericalDropletsElement.from_droplets(Emulsion([], dtype=d.data.dtype))
    check(el, 0)

    el = SphericalDropletsElement.empty(0, droplet=d)
    check(el, 0)

    el = SphericalDropletsElement.empty(0, dim=dim)
    check(el, 0)

    # additional space
    el = SphericalDropletsElement.from_droplets(
        Emulsion([], dtype=d.data.dtype), maxcount=3
    )
    check(el, 3)

    el = SphericalDropletsElement.empty(3, droplet=d)
    check(el, 3)

    el = SphericalDropletsElement.empty(3, dim=dim)
    check(el, 3)

    with pytest.raises(TypeError):
        SphericalDropletsElement.empty(1)
