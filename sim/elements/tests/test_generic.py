"""
Test generic elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid
from pde.tools.misc import skipUnlessModule

from .. import (
    ArrowsElement,
    MeanfieldElement,
    ObjectElementBase,
    PointsElement,
    ScalarFieldElement,
    SphericalDropletsElement,
)


class EmptyElement(ObjectElementBase):
    """dummy class to test the simplest element"""

    parameters_default = {"dim": 2}  # type: ignore

    @property
    def dim(self):
        return self.parameters["dim"]


def generate_elements(dim=None):
    """helper function generating all tested backgrounds"""
    yield EmptyElement(np.zeros(()), {"dim": 2 if dim is None else dim})

    if dim is None or dim == 1:
        yield PointsElement(np.random.randn(3, 1))
    if dim is None or dim == 2:
        yield PointsElement(np.random.randn(3, 2))

    if dim is None or dim == 1:
        yield ArrowsElement.from_position_random_direction(np.random.randn(3, 1))
    if dim is None or dim == 2:
        yield ArrowsElement.from_position_random_direction(np.random.randn(3, 2))

    if dim is None or dim == 1:
        emulsion = [SphericalDroplet([0], 1), SphericalDroplet([1], 2)]
        yield SphericalDropletsElement.from_droplets(emulsion)
    if dim is None or dim == 2:
        emulsion = [SphericalDroplet([0, 0], 1), SphericalDroplet([1, 1], 2)]
        yield SphericalDropletsElement.from_droplets(emulsion)

    if dim is None or dim == 1:
        yield MeanfieldElement(0.1, {"bounds": [[0, 1]]})  # 1d
    if dim is None or dim == 2:
        yield MeanfieldElement(0.1, {"bounds": [[0, 1], [0, 1]]})  # 2d

    if dim is None or dim == 2:
        field = ScalarField.random_normal(UnitGrid([3, 3]))
        yield ScalarFieldElement.from_field(field)


@pytest.mark.parametrize("element", generate_elements())
def test_basic(element):
    """test basic functions of elements"""
    assert isinstance(str(element), str)
    assert isinstance(repr(element), str)
    assert isinstance(element.attributes, dict)
    assert isinstance(element.data, np.ndarray)

    e1 = element.copy()
    assert e1 is not element
    assert e1 == element

    # test generic plotting
    if isinstance(element, PointsElement) or element.dim == 2:
        element.plot(action="close")


@skipUnlessModule("h5py")
@pytest.mark.parametrize("element", generate_elements())
def test_element_io(element, tmp_path):
    """test writing and reading element states"""
    path = tmp_path / f"test_io_{element.__class__.__name__}.hdf"

    element.to_file(path)
    element2 = ElementBase.from_file(path)
    assert element == element2
    assert element is not element2
