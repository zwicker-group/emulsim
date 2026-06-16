"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField, UnitGrid

from emulsim import Simulation, State
from emulsim.actors.autonomous import DiffusionActor
from emulsim.actors.coupling.fields import (
    FieldBoundaryExchangeActor,
    FieldCouplingActor,
    FieldExchangeActor,
)
from emulsim.elements import (
    MeanfieldElement,
    ScalarBoundaryFieldElement,
    ScalarFieldElement,
)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_fields_1(dim):
    """Simple test of single fields."""
    if dim == 0:
        grid = UnitGrid([3])
        element = MeanfieldElement(2, {"bounds": grid.axes_bounds})

    else:
        field = ScalarField(UnitGrid([3] * dim), 2)
        element = ScalarFieldElement.from_field(field)

    actor = FieldCouplingActor({"fields": ["a"], "evolution_rates": {"a": "1 + t"}})

    state = element.copy(method="data")
    actor.evolve((state,), 1, 2)
    assert np.allclose(state.data, 6)

    state = element.copy(method="data")
    evolver = actor.make_evolver_numba((state,))
    evolver((state._data_numba,), 1, 2)
    assert np.allclose(state.data, 6)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_fields_2(dim, rng):
    """Simple test of two fields."""
    if dim == 0:
        grid = UnitGrid([3])
        element1 = MeanfieldElement(2, {"bounds": grid.axes_bounds})
        element2 = MeanfieldElement(3, {"bounds": grid.axes_bounds})

    else:
        grid = UnitGrid([3] * dim)
        element1 = ScalarFieldElement.from_field(ScalarField(grid, 2))
        field = ScalarField.random_uniform(grid, rng=rng)
        element2 = ScalarFieldElement.from_field(field)

    actor = FieldCouplingActor({"fields": ["a", "b"], "evolution_rates": {"a": "+b"}})

    e1 = element1.copy(method="data")
    e2 = element2.copy(method="data")
    actor.evolve((e1, e2), 0, 1)
    assert np.allclose(e1.data, element1.data + element2.data)
    assert np.allclose(e2.data, element2.data)

    e1 = element1.copy(method="data")
    e2 = element2.copy(method="data")
    evolver = actor.make_evolver_numba((e1, e2))
    evolver((e1._data_numba, e2._data_numba), 0, 1)
    assert np.allclose(e1.data, element1.data + element2.data)
    assert np.allclose(e2.data, element2.data)


@pytest.mark.parametrize("resolution", [4, 2])
def test_field_boundary_coupling_resolution(resolution, rng):
    """Simple test of the boundary coupling."""
    # set up state
    grid = UnitGrid([4, 4], periodic=True)
    grid_bulk = CartesianGrid(grid.axes_bounds, [resolution, 4], periodic=True)
    bulk = ScalarFieldElement.from_field(ScalarField(grid_bulk, 0.001))
    bndry = ScalarBoundaryFieldElement.from_bulk_grid(
        grid, axis=1, upper=True, data=rng.normal(size=4)
    )
    state = State({"bulk": bulk, "bndry": bndry})
    total_amount = state.get_total_quantity("total_amount")

    # set up simulation
    simulation = Simulation(state)
    if resolution > 1:
        simulation.add_actor("bulk", DiffusionActor())
    flux = "0.1 * (bulk - boundary)"
    boundary_coupling = FieldBoundaryExchangeActor({"exchange_flux": flux})
    simulation.add_actor(("bulk", "bndry"), boundary_coupling)

    res1 = simulation.run(t_range=1, dt=0.01, backend="numpy", tracker=None)
    res2 = simulation.run(t_range=1, dt=0.01, backend="numba", tracker=None)

    np.testing.assert_allclose(res1["bulk"].data, res2["bulk"].data)
    np.testing.assert_allclose(res1["bndry"].data, res2["bndry"].data)
    assert pytest.approx(total_amount) == res1.get_total_quantity("total_amount")
    assert pytest.approx(total_amount) == res2.get_total_quantity("total_amount")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_field_boundary_coupling_meanfield(backend, rng):
    """Simple test of the boundary coupling for mean field simulations."""
    # set up state
    grid_full = UnitGrid([4, 4], periodic=True)
    grid_mean = CartesianGrid(grid_full.axes_bounds, [1, 1], periodic=True)
    field = ScalarField(grid_mean, 0.001)
    bulk_full = ScalarFieldElement.from_field(field.copy())
    bulk_mean = MeanfieldElement.from_field(field.copy())
    assert bulk_full.volume == 16
    assert bulk_mean.volume == 16

    bndry_full = ScalarBoundaryFieldElement.from_bulk_grid(
        grid_full, axis=1, upper=True, data=rng.normal(size=4)
    )
    bndry_mean = bndry_full.copy()
    state_full = State({"bulk": bulk_full, "bndry": bndry_full})
    state_mean = State({"bulk": bulk_mean, "bndry": bndry_mean})
    total_amount = state_full.get_total_quantity("total_amount")
    assert total_amount == state_mean.get_total_quantity("total_amount")

    # set up simulation
    simulation_full = Simulation(state_full)
    simulation_full.add_actor("bulk", DiffusionActor())
    simulation_mean = Simulation(state_full)
    flux = "0.1 * (bulk - boundary)"
    boundary_coupling_full = FieldBoundaryExchangeActor({"exchange_flux": flux})
    boundary_coupling_mean = boundary_coupling_full.copy()
    simulation_full.add_actor(("bulk", "bndry"), boundary_coupling_full)
    simulation_mean.add_actor(("bulk", "bndry"), boundary_coupling_mean)

    res_full = simulation_full.run(t_range=1, dt=0.01, backend=backend, tracker=None)
    res_mean = simulation_mean.run(t_range=1, dt=0.01, backend=backend, tracker=None)

    print(simulation_full.diagnostics["solver"]["steps"] == 100)
    print(simulation_mean.diagnostics["solver"]["steps"] == 100)

    assert boundary_coupling_full._cache["grid_match"] == "boundary_resolved"
    assert boundary_coupling_mean._cache["grid_match"] == "boundary_resolved"

    np.testing.assert_allclose(res_full["bulk"].data, res_mean["bulk"].data)
    np.testing.assert_allclose(res_full["bndry"].data, res_mean["bndry"].data)
    assert pytest.approx(total_amount) == res_full.get_total_quantity("total_amount")
    assert pytest.approx(total_amount) == res_mean.get_total_quantity("total_amount")


@pytest.mark.parametrize(
    "cls1,cls2",
    [
        (MeanfieldElement, ScalarFieldElement),
        (ScalarFieldElement, MeanfieldElement),
        (ScalarFieldElement, ScalarFieldElement),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_field_exchange_actor(cls1, cls2, backend):
    """Test the FieldExchangeActor."""
    grid = UnitGrid([2, 2])

    field1 = cls1.from_field(ScalarField(grid, 2))
    field2 = cls2.from_field(ScalarField(grid, 1))
    actor = FieldExchangeActor({"exchange_rate": "c1 - c2"})

    if backend == "numpy":
        actor.evolve((field1, field2), 0, 2)
    elif backend == "numba":
        evolver = actor.make_evolver_numba((field1, field2))
        evolver((field1._data_numba, field2._data_numba), 0, 2)
    else:
        raise ValueError

    np.testing.assert_allclose(field1.data, 0)
    np.testing.assert_allclose(field2.data, 3)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_field_exchange_actor_different_volume(backend):
    """Test the FieldExchangeActor with bulks of different volume."""
    grid = CartesianGrid([(0, 1), (0, 1)], 2)

    field1 = ScalarFieldElement.from_field(ScalarField(grid, 2))
    field2 = MeanfieldElement.from_field(ScalarField(grid, 1), {"volume": 4})
    actor = FieldExchangeActor({"exchange_rate": "c1 - c2"})

    if backend == "numpy":
        actor.evolve((field1, field2), 0, 2)
    elif backend == "numba":
        evolver = actor.make_evolver_numba((field1, field2))
        evolver((field1._data_numba, field2._data_numba), 0, 2)
    else:
        raise ValueError

    np.testing.assert_allclose(field1.data, 0)
    np.testing.assert_allclose(field2.data, 1.5)
