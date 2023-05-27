"""
Test generic elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection, ScalarField, UnitGrid

from sim.elements import (
    FieldCollectionElement,
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

    # test basic error estimate
    for backend in ["numpy", "numba"]:
        error_estimator = element._make_error_estimator(backend=backend)
        assert error_estimator(element._data_numba, element._data_numba) == 0


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


@pytest.mark.parametrize(
    "element",
    [
        MeanfieldElement(1, {"bounds": [[0, 3]]}),
        MeanfieldElement.from_field(ScalarField(UnitGrid([3]), 1)),
    ],
)
def test_meanfield_basic(element):
    """test basic methods of the simple mean field element"""
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


@pytest.mark.parametrize("axis", [1, -1])
def test_boundaryfield(axis):
    """test basic methods of the simple boundary field element"""
    grid = UnitGrid([10, 8])
    element = ScalarBoundaryFieldElement.from_bulk_grid(
        grid, axis=axis, upper=True, data=1
    )

    try:
        assert element.grid == grid.slice((0,))
    except AttributeError:
        # fall-back for deprecated method (remove on 2023-03-15)
        assert element.grid == grid.get_subgrid((0,))
    np.testing.assert_array_equal(element.data, 1)
    assert element.degrees_of_freedom == 10
    assert element.axis == 1

    bulk_coords = element.bulk_coordinates
    np.testing.assert_allclose(bulk_coords[:, 0], grid.cell_coords[:, 0, 0])
    np.testing.assert_allclose(bulk_coords[:, 1], 8)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_fields", [1, 2])
def test_field_collection(dim, num_fields):
    """test basic methods of the FieldCollection"""
    grid = UnitGrid([5] * dim)
    fc = FieldCollection.scalar_random_uniform(num_fields, grid)
    element = FieldCollectionElement.from_fields(fc)

    assert element.dim == dim
    assert element.num_fields == num_fields
    np.testing.assert_allclose(element.amounts, fc.integrals)

    # test numpy functions
    element.data[...] = 0
    assert element.total_amount == 0
    amounts = np.arange(num_fields) + 1.5
    element.add_amounts([1] * dim, amounts)
    np.testing.assert_allclose(element.amounts, amounts)
    conc = element.get_concentrations([3] * dim)
    np.testing.assert_allclose(conc, 0)

    # test numba functions
    element.data[...] = 0
    assert element.total_amount == 0
    adder = element.make_add_amounts_compiled()
    adder(element.data, np.ones(dim), amounts)
    np.testing.assert_allclose(element.amounts, amounts)
    getter = element.make_get_concentrations_compiled()
    np.testing.assert_allclose(getter(element.data, np.full(dim, 3)), 0)
