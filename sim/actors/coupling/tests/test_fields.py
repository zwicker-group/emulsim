"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np
import pytest

from pde import ScalarField, UnitGrid

from .... import Simulation, State
from ....elements import (
    MeanfieldElement,
    ScalarBoundaryFieldElement,
    ScalarFieldElement,
)
from ...autonomous import DiffusionActor
from ..fields import FieldBoundaryExchangeActor, FieldCouplingActor


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_fields_1(dim):
    """simple test of single fields"""
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
    """simple test of two fields"""
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


def test_fields_boundary_coupling():
    """simple test of the boundary coupling"""

    # set up state
    grid = UnitGrid([4, 4], periodic=True)
    bulk = ScalarFieldElement.from_field(ScalarField(grid, 0.001))
    data = np.random.randn(4)
    bndry = ScalarBoundaryFieldElement.from_bulk_grid(
        grid, axis=1, upper=True, data=data
    )
    state = State({"bulk": bulk, "bndry": bndry})
    total_amount = state.get_total_quantity("total_amount")

    # set up simulation
    simulation = Simulation(state)
    simulation.add_actor("bulk", DiffusionActor())
    boundary_coupling = FieldBoundaryExchangeActor({"exchange_flux": "bulk - boundary"})
    simulation.add_actor(("bulk", "bndry"), boundary_coupling)

    res1 = simulation.run(t_range=1, backend="numpy", tracker=None)
    res2 = simulation.run(t_range=1, backend="numba", tracker=None)

    np.testing.assert_allclose(res1["bulk"].data, res2["bulk"].data)
    np.testing.assert_allclose(res1["bndry"].data, res2["bndry"].data)
    assert pytest.approx(total_amount) == res1.get_total_quantity("total_amount")
    assert pytest.approx(total_amount) == res2.get_total_quantity("total_amount")
