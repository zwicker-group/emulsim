"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
.. codeauthor:: Ajinkya Kulkarni <ajinkya.kulkarni@ds.mpg.de>
"""


import numpy as np
import pytest

from droplets import Emulsion, SphericalDroplet
from pde import CartesianGrid, ScalarField, UnitGrid
from pde.grids.base import DimensionError
from pde.tools.misc import skipUnlessModule

from .... import ReactionDiffusionActor, Simulation, State
from ....elements import MeanfieldElement, ScalarFieldElement, SphericalDropletsElement
from ..spherical_droplet import ShellCollection, SphericalDropletActor


def recarrays_allclose(a, b):
    """tests whether the entries of two structured arrays are all close"""
    if a.dtype != b.dtype:
        return False
    return all(np.allclose(a[name], b[name]) for name in a.dtype.names)


def test_shells_1d():
    """test shell collection in 1 dimensions"""
    sc = ShellCollection.generate(dim=1)
    assert len(sc) == 1
    shell = sc.get_shell(1e3)
    assert shell.vectors.shape == (2, 1)
    assert shell.weights.shape == (2,)
    np.testing.assert_allclose(shell.weights, np.full(2, 0.5))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_shells_general(dim):
    """test shell collections in 2 and 3 dimensions"""
    sc = ShellCollection.generate(dim=dim)

    for shell in sc:
        vs, ws = shell.vectors, shell.weights
        assert vs.shape[1] == dim
        assert vs.shape[0] == len(ws)
        assert ws.sum() == pytest.approx(1)
        np.testing.assert_allclose(ws @ vs, np.zeros(dim), atol=1e-10)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets(dim):
    """simple test of SphericalDropletAgents"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == 1

    coupling = SphericalDropletActor()
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
    evolver((droplets.data, field.data), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    assert droplets.data[0].radius == radius

    droplets2 = droplets.copy()
    assert droplets2 is not droplets
    assert np.array_equal(droplets2.data, droplets.data)

    # test whether plotting works in principle
    if dim == 2:
        coupling.plot_shell_points((droplets, field))

    # test incompatible dimensions
    droplet_dim = (None, 2, 1, 1)[dim]
    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([1] * droplet_dim, 1)]
    )
    coupling = SphericalDropletActor()
    with pytest.raises(DimensionError):
        coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_const_shell_count(dim):
    """simple test of SphericalDropletAgents"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == 1

    coupling = SphericalDropletActor(
        {"shell_sector_method": "count", "shell_sector_count": 2 * dim}
    )
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
    evolver((droplets.data, field.data), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    assert droplets.data[0].radius == radius

    droplets2 = droplets.copy()
    assert droplets2 is not droplets
    assert np.array_equal(droplets2.data, droplets.data)

    # test whether plotting works in principle
    if dim == 2:
        coupling.plot_shell_points((droplets, field))

    # test incompatible dimensions
    droplet_dim = (None, 2, 1, 1)[dim]
    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([1] * droplet_dim, 1)]
    )
    coupling = SphericalDropletActor()
    with pytest.raises(DimensionError):
        coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_reactions_inside(dim, compiled):
    """simple test of SphericalDropletAgents with reactions"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    d1 = SphericalDropletsElement.from_droplets([SphericalDroplet([1] * dim, 1)])
    c1 = SphericalDropletActor()

    d2 = SphericalDropletsElement.from_droplets([SphericalDroplet([2] * dim, 1)])
    c2 = SphericalDropletActor({"reaction_inside": "-1"})

    state = State({"field": field, "d1": d1, "d2": d2})
    sim = Simulation(state)
    sim.add_actor(("d1", "field"), c1)
    sim.add_actor(("d2", "field"), c2)

    assert 0 < sim.estimate_dt(state) < 1000

    if compiled:
        evolver = sim.make_evolver_numba(state)
        evolver(state.data, 0, 0.5)
    else:
        sim.evolve(state, 0, 0.5)
    assert d1.total_amount > d2.total_amount


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_reactions_outside(dim, compiled):
    """simple test of SphericalDropletAgents with reactions"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    d1 = SphericalDropletsElement.from_droplets([SphericalDroplet([1] * dim, 1)])
    c1 = SphericalDropletActor()

    d2 = SphericalDropletsElement.from_droplets([SphericalDroplet([2] * dim, 1)])
    c2 = SphericalDropletActor({"reaction_outside": "-1"})

    state = State({"field": field, "d1": d1, "d2": d2})
    sim = Simulation(state)
    sim.add_actor(("d1", "field"), c1)
    sim.add_actor(("d2", "field"), c2)

    assert 0 < sim.estimate_dt(state) < 1000

    if compiled:
        evolver = sim.make_evolver_numba(state)
        evolver(state.data, 0, 0.5)
    else:
        sim.evolve(state, 0, 0.5)
    assert d1.total_amount > d2.total_amount


@skipUnlessModule("phasesep")
@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_linearized_fluxes(dim, backend):
    """a simple test for implementation of linearized fluxes for Active Emulsions under
    mean-field conditions"""
    # Make meanfield grid
    grid = CartesianGrid([(0, 1000)] * dim, 1, periodic=True)
    background = ScalarFieldElement.from_field(ScalarField(grid, 0.005))

    # Initialize radius and positions of 100 droplets
    droplet_data = [
        SphericalDroplet(
            position=grid.get_random_point(), radius=np.random.uniform(4, 6)
        )
        for _ in range(100)
    ]
    droplets = SphericalDropletsElement.from_droplets(droplet_data)
    state = State({"background": background, "droplets": droplets})

    # Initialize simulation
    reaction_flux = "0.001 - 0.01 * c"
    simulation = Simulation(state)
    simulation.add_actor(
        "background", ReactionDiffusionActor({"reaction_flux": reaction_flux})
    )
    droplet_actor = SphericalDropletActor(
        {
            "shell_thickness": grid.discretization[0],
            "shell_sector_size": grid.discretization[0],
            "reaction_inside": -0.01,
            "reaction_outside": reaction_flux,
        }
    )
    simulation.add_actor(("droplets", "background"), droplet_actor)

    # Run simulation
    result = simulation.run(t_range=int(1e3), backend=backend)

    # Check if droplet radii within the second decimal place
    final_droplet_radii = result["droplets"].data["radius"]
    np.testing.assert_allclose(
        final_droplet_radii, final_droplet_radii.mean(), rtol=0.1, atol=0
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_coarsening(dim):
    """simple test of coarsening"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    emulsion = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(), 0.1),
            SphericalDroplet(grid.get_random_point(), 0.2),
        ]
    )
    droplets = SphericalDropletsElement.from_droplets(emulsion)

    coupling = SphericalDropletActor()

    ceq = coupling.get_equilibrium_concentrations(droplets).mean()
    field.concentration = ceq

    total_amount = pytest.approx(field.total_amount + droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.1)
    assert field.total_amount + droplets.total_amount == total_amount

    assert droplets.data[0].radius < 0.1
    assert droplets.data[1].radius > 0.2


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_drift(dim, backend):
    """test drift direction of droplets"""
    grid = UnitGrid([4] * dim)
    # initialize gradient along x-direction
    field_data = ScalarField.from_expression(grid, "x / 40")

    for drift in [True, False]:
        field = ScalarFieldElement.from_field(field_data)
        droplets = SphericalDropletsElement.from_droplets(
            [SphericalDroplet([2] * dim, 0.2)]
        )
        state = State({"droplets": droplets, "field": field})

        coupling = SphericalDropletActor({"drift_enabled": drift})

        sim = Simulation(state)
        sim.add_actor(("droplets", "field"), coupling)
        res = sim.run(t_range=1, backend=backend)

        d = res["droplets"]

        assert np.all(d.data["radius"] > 0.2)

        if drift:
            assert d.data["position"][0, 0] > 2
            np.testing.assert_allclose(
                d.data["position"][:, 1:], np.full((1, dim - 1), 2), rtol=1e-2
            )
        else:
            np.testing.assert_allclose(
                d.data["position"], np.full((1, dim), 2), rtol=1e-2
            )


def test_multithreading():
    """simple consistency test for multiprocessing"""
    grid = UnitGrid([1])
    field1 = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    field2 = field1.copy()

    emulsion = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(), np.random.uniform(0.01, 0.02))
            for _ in range(100)
        ]
    )
    droplets1 = SphericalDropletsElement.from_droplets(emulsion)
    droplets2 = SphericalDropletsElement.from_droplets(emulsion)

    coupling1 = SphericalDropletActor({"num_threads": 1})
    coupling2 = SphericalDropletActor({"num_threads": 2})

    evolver1 = coupling1.make_evolver_numba((droplets1, field1))
    evolver2 = coupling2.make_evolver_numba((droplets2, field2))

    evolver1((droplets1.data, field1.data), 0, 0.001)
    evolver2((droplets2.data, field2.data), 0, 0.001)

    np.testing.assert_allclose(field1.data, field2.data, rtol=0.1)
