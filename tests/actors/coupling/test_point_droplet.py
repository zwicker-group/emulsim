"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured

from droplets import Emulsion, SphericalDroplet
from pde.grids import UnitGrid

from sim import Simulation, State
from sim.actors.coupling.point_droplet import PointDropletActor
from sim.elements import MeanfieldElement, SphericalDropletsElement


@pytest.mark.parametrize("dim", [3])
def test_point_droplets_diffusion(dim):
    """simple test of point droplets with diffusive exchange"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == len(droplets) == 1

    coupling = PointDropletActor()
    assert isinstance(coupling.info, dict)
    assert coupling.num_elements == 2

    assert 0 < coupling.estimate_dt((droplets, field)) < 1000
    total_amount = pytest.approx(droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    radius = pytest.approx(droplets.data[0].radius)

    evolver = coupling.make_evolver_numba((droplets, field))
    droplets.data[0].radius = 1  # reset radius to check whether it agrees
    field.concentration = 0
    evolver((droplets._data_numba, field._data_numba), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    assert droplets.data[0].radius == radius

    droplets2 = droplets.copy(method="data")
    assert droplets2 is not droplets
    assert np.array_equal(droplets2.data, droplets.data)

    # test incompatible dimensions
    droplets = SphericalDropletsElement.from_droplets([SphericalDroplet([1], 1)])
    coupling = PointDropletActor()
    with pytest.raises(NotImplementedError):
        coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("dim", [3])
def test_point_droplets_diffusion_coarsening(dim):
    """simple test of coarsening with diffusive exchange"""
    grid = UnitGrid([3] * dim)
    field1 = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field1.concentration == pytest.approx(0)

    emulsion1 = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(), 0.1),
            SphericalDroplet(grid.get_random_point(), 0.2),
        ]
    )
    droplets1 = SphericalDropletsElement.from_droplets(emulsion1)
    droplets2 = SphericalDropletsElement.from_droplets(emulsion1, copy=True)
    assert droplets1.droplet_count == 2

    coupling1 = PointDropletActor()
    ceq = coupling1.get_equilibrium_concentrations(droplets1).mean()
    field1.concentration = ceq
    field2 = field1.copy(method="data")

    total_amount = pytest.approx(field1.total_amount + droplets1.total_amount)

    coupling1.evolve((droplets1, field1), 0, 0.1)
    assert field1.total_amount + droplets1.total_amount == total_amount
    assert emulsion1[0].radius < 0.1
    assert emulsion1[1].radius > 0.2

    coupling2 = PointDropletActor(
        {"flux_model": "linear", "exchange_rate": "4 * pi * R"}
    )
    coupling2.evolve((droplets2, field2), 0, 0.1)
    assert field2.total_amount + droplets2.total_amount == total_amount
    np.testing.assert_allclose(
        structured_to_unstructured(droplets1.data),
        structured_to_unstructured(droplets2.data),
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_material_conservation(backend):
    """test whether the simulation conserves the total amount of material"""
    grid = UnitGrid([4] * 3, periodic=True)
    field = MeanfieldElement(1, {"bounds": grid.axes_bounds})

    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([2] * 3, 0.5)], parameters={"droplet_concentration": 3}
    )
    state = State({"droplets": droplets, "field": field})
    total_amount = state.get_total_quantity("total_amount")

    coupling = PointDropletActor({"equilibrium_concentration": "1"})

    sim = Simulation(state)
    sim.add_actor(("droplets", "field"), coupling)
    res = sim.run(t_range=10, backend=backend)

    assert res.get_total_quantity("total_amount") == pytest.approx(total_amount)


@pytest.mark.parametrize("dim", [1, 2])
def test_point_droplets_linear(dim):
    """simple test of point droplets with linear exchange"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == len(droplets) == 1

    coupling = PointDropletActor({"flux_model": "linear"})
    assert isinstance(coupling.info, dict)
    assert coupling.num_elements == 2

    assert 0 < coupling.estimate_dt((droplets, field)) < 1000
    total_amount = pytest.approx(droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    radius = pytest.approx(droplets.data[0].radius)

    evolver = coupling.make_evolver_numba((droplets, field))
    droplets.data[0].radius = 1  # reset radius to check whether it agrees
    field.concentration = 0
    evolver((droplets._data_numba, field._data_numba), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    assert droplets.data[0].radius == radius

    droplets2 = droplets.copy(method="data")
    assert droplets2 is not droplets
    assert np.array_equal(droplets2.data, droplets.data)


@pytest.mark.parametrize("dim", [1, 2])
def test_point_droplets_linear_coarsening(dim):
    """simple test of coarsening with linear exchange"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    emulsion = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(), 0.1),
            SphericalDroplet(grid.get_random_point(), 0.2),
        ]
    )
    droplets = SphericalDropletsElement.from_droplets(emulsion)
    assert droplets.droplet_count == 2

    coupling = PointDropletActor({"flux_model": "linear"})

    ceq = coupling.get_equilibrium_concentrations(droplets).mean()
    field.concentration = ceq

    total_amount = pytest.approx(field.total_amount + droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.1)
    assert field.total_amount + droplets.total_amount == total_amount

    assert emulsion[0].radius < 0.1
    assert emulsion[1].radius > 0.2


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_point_droplets_reactions_inside(dim, compiled):
    """simple test of SphericalDropletAgents with reactions"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    d1 = SphericalDropletsElement.from_droplets([SphericalDroplet([1] * dim, 1)])
    c1 = PointDropletActor({"flux_model": "diffusion" if dim == 3 else "linear"})

    d2 = SphericalDropletsElement.from_droplets([SphericalDroplet([2] * dim, 1)])
    c2 = PointDropletActor(
        {
            "flux_model": "diffusion" if dim == 3 else "linear",
            "mean_reaction_inside": "-1",
        }
    )

    state = State({"field": field, "d1": d1, "d2": d2})
    sim = Simulation(state)
    sim.add_actor(("d1", "field"), c1)
    sim.add_actor(("d2", "field"), c2)

    assert 0 < sim.estimate_dt(state) < 1000

    if compiled:
        evolver = sim.make_evolver_numba(state)
        evolver(state._data_numba, 0, 0.5)
    else:
        sim.evolve(state, 0, 0.5)
    assert d1.total_amount > d2.total_amount
