"""
Test spherical droplets elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from droplets import SphericalDroplet

from .. import SphericalDropletsElement


@pytest.mark.parametrize("dim", [1, 2])
def test_spherical_droplets(dim):
    """ test basic spherical droplets functions """
    emulsion = [SphericalDroplet([0] * dim, 1), SphericalDroplet([1] * dim, 2)]
    element = SphericalDropletsElement.from_droplets(emulsion)
    assert element.dim == dim
    assert element.degrees_of_freedom == 2 * (dim + 1)
