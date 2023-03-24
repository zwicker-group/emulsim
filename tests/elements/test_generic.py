"""
Test generic elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy

import numpy as np
import pytest

from droplets import SphericalDroplet
from pde import FieldCollection, ScalarField, UnitGrid
from pde.tools.numba import jit

from sim.elements import (
    ArrowsElement,
    FieldCollectionElement,
    MeanfieldElement,
    MulticomponentDroplet,
    MulticomponentDropletsElement,
    ObjectElementBase,
    PointsElement,
    ScalarBoundaryFieldElement,
    ScalarFieldElement,
    SphericalDropletsElement,
)
from sim.elements.base import _ElementBase


class EmptyElement(ObjectElementBase):
    """dummy class to test the simplest element"""

    parameters_default = {"dim": 2}  # type: ignore

    def __init__(self, data=None, parameters=None):
        assert data is None
        super().__init__(None, parameters)

    @property
    def dim(self):
        return self.parameters["dim"]


def generate_elements(dim=None, incl_obj=True):
    """helper function generating all tested backgrounds"""
    if incl_obj:
        yield EmptyElement(parameters={"dim": 2 if dim is None else dim})

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
        emulsion = [
            MulticomponentDroplet([0], 1, [2, 3]),
            MulticomponentDroplet([1], 2, [0, 1]),
        ]
        yield MulticomponentDropletsElement.from_droplets(emulsion)
    if dim is None or dim == 2:
        emulsion = [
            MulticomponentDroplet([0, 0], 1, [2, 2, 2]),
            MulticomponentDroplet([1, 1], 2, [3, 3, 3]),
        ]
        yield MulticomponentDropletsElement.from_droplets(emulsion)

    if dim is None or dim == 1:
        yield MeanfieldElement(0.1, {"bounds": [[0, 1]]})  # 1d
    if dim is None or dim == 2:
        yield MeanfieldElement(0.1, {"bounds": [[0, 1], [0, 1]]})  # 2d

    if dim is None or dim == 2:
        field = ScalarField.random_normal(UnitGrid([3, 3]))
        yield ScalarFieldElement.from_field(field)

    if dim is None or dim == 2:
        fields = FieldCollection.scalar_random_uniform(2, UnitGrid([3, 3]))
        yield FieldCollectionElement.from_fields(fields)

    if dim is None or dim == 2:
        grid = UnitGrid([3, 3])
        yield ScalarBoundaryFieldElement.from_bulk_grid(
            grid, axis=1, upper=True, data=1, parameters={"label": "boundary_field"}
        )


@pytest.mark.parametrize("element", generate_elements())
def test_basic(element):
    """test basic functions of elements"""
    assert isinstance(str(element), str)
    assert isinstance(repr(element), str)
    assert isinstance(element.attributes, dict)

    e1 = element.copy()
    assert e1 is not element
    assert e1 == element

    e2 = copy.copy(element)
    assert e2 is not element
    assert e2 == element

    # test generic plotting
    if isinstance(element, PointsElement) or element.dim == 2:
        element.plot(action="close")


@pytest.mark.parametrize("element", generate_elements())
def test_numba_data_access(element, capsys):
    """test whether the element data can be used in numba"""

    @jit
    def printer(element_data):
        print(element_data)

    printer(element._data_numba)
    captured = capsys.readouterr()
    assert captured.out != ""


@pytest.mark.parametrize("element", generate_elements())
@pytest.mark.parametrize("ext", ["zarr", "json"])
def test_element_io(element, ext, tmp_path):
    """test writing and reading element states"""
    path = tmp_path / f"test_io_{element.__class__.__name__}.{ext}"

    element.to_file(path)
    element2 = _ElementBase.from_file(path)
    assert element == element2
    assert element is not element2
