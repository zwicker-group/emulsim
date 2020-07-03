"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import pytest
import numpy as np

from pde import UnitGrid, ScalarField

from ..fields import FieldCouplingActor
from ....elements import MeanfieldElement, ScalarFieldElement


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_fields_1(dim):
    """ simple test of single fields """
    if dim == 0:
        grid = UnitGrid([3])
        element = MeanfieldElement(2, {"bounds": grid.axes_bounds})

    else:
        field = ScalarField(UnitGrid([3] * dim), 2)
        element = ScalarFieldElement.from_field(field)

    actor = FieldCouplingActor({"fields": ["a"], "evolution_rates": {"a": "1 + t"}})

    state = element.copy()
    actor.evolve((state,), 1, 2)
    assert np.allclose(state.data, 6)

    state = element.copy()
    evolver = actor.make_evolver_numba((state,))
    evolver((state.data,), 1, 2)
    assert np.allclose(state.data, 6)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_fields_2(dim):
    """ simple test of two fields """
    if dim == 0:
        grid = UnitGrid([3])
        element1 = MeanfieldElement(2, {"bounds": grid.axes_bounds})
        element2 = MeanfieldElement(3, {"bounds": grid.axes_bounds})

    else:
        grid = UnitGrid([3] * dim)
        element1 = ScalarFieldElement.from_field(ScalarField(grid, 2))
        element2 = ScalarFieldElement.from_field(ScalarField.random_uniform(grid))

    actor = FieldCouplingActor({"fields": ["a", "b"], "evolution_rates": {"a": "+b"}})

    e1 = element1.copy()
    e2 = element2.copy()
    actor.evolve((e1, e2), 0, 1)
    assert np.allclose(e1.data, element1.data + element2.data)
    assert np.allclose(e2.data, element2.data)

    e1 = element1.copy()
    e2 = element2.copy()
    evolver = actor.make_evolver_numba((e1, e2))
    evolver((e1.data, e2.data), 0, 1)
    assert np.allclose(e1.data, element1.data + element2.data)
    assert np.allclose(e2.data, element2.data)
