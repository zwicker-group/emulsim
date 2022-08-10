"""
Test generic elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField, UnitGrid

from .. import (
    MeanfieldElement,
    ReservoirElement,
    ScalarBoundaryFieldElement,
    ScalarFieldElement,
)


def generate_field_elements():
    """helper function generating all tested field elements"""
    grid = CartesianGrid([[0, 3], [0, 3]], 1)
    element = MeanfieldElement(parameters={"bounds": grid.axes_bounds})
    yield element, grid

    grid = UnitGrid([3, 3])
    element = ScalarFieldElement(parameters={"grid": grid})
    yield element, grid


@pytest.mark.parametrize("element,grid", generate_field_elements())
def test_elements_numpy(element, grid):
    """test functions based on numpy"""
    assert isinstance(element.attributes, dict)

    # test adding amounts
    assert element.total_amount == 0
    p = grid.get_random_point()
    element.add_amount(p, 3)
    assert element.total_amount == pytest.approx(3)

    # set data to a fixed value
    element.data[:] = 2
    assert element.get_concentration(p) == pytest.approx(2)
    assert np.allclose(element.get_concentration([p, p]), np.full(2, 2))

    # test copying the element
    element2 = element.copy()
    assert element2 is not element
    assert np.array_equal(element2.data, element.data)


@pytest.mark.parametrize("element,grid", generate_field_elements())
def test_elements_numba(element, grid):
    """test functions based on numba"""
    # test adding amounts
    assert element.total_amount == 0
    p = grid.get_random_point()
    adder = element.make_add_amount_compiled()
    adder(element.data, p, 3)
    assert element.total_amount == pytest.approx(3)

    # set data to a fixed value
    element.data[:] = 2
    getter = element.make_get_concentration_compiled()
    assert getter(element.data, p) == pytest.approx(2)
    result = element.get_concentration(np.c_[p, p].T)
    assert np.allclose(result, np.full(2, 2))


def test_reservoir():
    """test the ReservoirElement"""
    element = ReservoirElement()

    assert isinstance(element.attributes, dict)

    # test adding amounts
    assert element.concentration == 0
    element.add_amount([0, 0], 3)
    assert element.concentration == 0

    adder = element.make_add_amount_compiled()
    adder(element.data, np.zeros(3), 3)
    assert element.concentration == 0

    # set data to a fixed value
    element.data[:] = 2
    p = np.zeros(2)
    assert element.get_concentration(p) == pytest.approx(2)
    assert np.allclose(element.get_concentration([p, p]), np.full(2, 2))

    getter = element.make_get_concentration_compiled()
    assert getter(element.data, p) == pytest.approx(2)
    result = element.get_concentration(np.c_[p, p].T)
    assert np.allclose(result, np.full(2, 2))

    # test copying the element
    element2 = element.copy()
    assert element2 is not element
    assert np.array_equal(element2.data, element.data)


def test_meanfield_basic():
    """test basic methods of the simple mean field element"""
    element = MeanfieldElement(1, {"bounds": [[0, 3]]})
    assert element.concentration == 1
    element.concentration = 2
    assert element.concentration == 2
    assert element.total_amount == 6
    assert element.grid == CartesianGrid([[0, 3]], 1)
    assert element.degrees_of_freedom == 1


def test_scalarfield():
    """test basic methods of the simple scalar element"""
    grid = UnitGrid([10])
    element = ScalarFieldElement(1, {"grid": grid})
    assert element.grid == grid
    np.testing.assert_array_equal(element.data, 1)
    assert element.degrees_of_freedom == 10


def test_boundaryfield():
    """test basic methods of the simple boundary field element"""
    grid = UnitGrid([10, 8])
    field = ScalarField(grid)
    element = ScalarBoundaryFieldElement.from_domain(field, axis=-1, data=1)

    assert element.grid == grid.get_subgrid((0,))
    np.testing.assert_array_equal(element.data, 1)
    assert element.degrees_of_freedom == 10
    assert element.axis == 1
