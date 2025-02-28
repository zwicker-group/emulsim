"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
.. codeauthor:: Ajinkya Kulkarni <ajinkya.kulkarni@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import spatial

from droplets import Emulsion, SphericalDroplet
from pde import CartesianGrid, ScalarField, UnitGrid
from pde.tools.misc import module_available

from helpers import assert_recarrays_allclose
from sim import ReactionDiffusionActor, Simulation, State
from sim.actors.coupling import spherical_droplet
from sim.elements import MeanfieldElement, ScalarFieldElement, SphericalDropletsElement


def test_spherical_polygon_area():
    """Test the function get_spherical_polygon_area."""
    area = spherical_droplet.get_spherical_polygon_area
    sector_area = area([[0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    assert sector_area == pytest.approx(np.pi / 2)


def test_spherical_voronoi(rng):
    """Test spatial.SphericalVoronoi."""
    # random points on the sphere
    ps = rng.random((32, 3)) - 0.5
    ps /= np.linalg.norm(ps, axis=1)[:, None]

    voronoi = spatial.SphericalVoronoi(ps)
    voronoi.sort_vertices_of_regions()

    total = sum(
        spherical_droplet.get_spherical_polygon_area(voronoi.vertices[reg])
        for reg in voronoi.regions
    )
    assert total == pytest.approx(4 * np.pi)


@pytest.mark.parametrize("dim", range(1, 4))
def test_points_on_sphere(dim, tmp_path):
    """Test spatial.SphericalVoronoi."""
    shell = spherical_droplet.PointsOnSphere.make_uniform(dim=dim)
    assert shell.dim == dim

    for balance_axes in [True, False]:
        ws = shell.get_area_weights(balance_axes=balance_axes)
        assert ws.sum() == pytest.approx(1)
        np.testing.assert_allclose(ws, 1 / len(shell.points), rtol=0.1)

    ws = shell.get_area_weights(balance_axes=True)
    np.testing.assert_allclose(ws @ shell.points, 0, atol=1e-15)

    path = tmp_path / f"test_points_on_sphere_{dim}.xyz"
    shell.write_to_xyz(path=path)
    assert path.stat().st_size > 0


def test_points_on_sphere_2(rng):
    """Special tests for 2 dimensions."""
    num = rng.integers(3, 9)
    shell = spherical_droplet.PointsOnSphere.make_uniform(dim=2, num_points=num)
    assert num * shell.get_mean_separation() == pytest.approx(2 * np.pi)


def test_shells_1d():
    """Test shell collection in 1 dimensions."""
    sc = spherical_droplet.ShellCollection.generate(dim=1)
    assert len(sc) == 1
    shell = sc.get_shell(1e3)
    assert shell.vectors.shape == (2, 1)
    assert shell.weights.shape == (2,)
    np.testing.assert_allclose(shell.weights, np.full(2, 0.5))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_shells_general(dim):
    """Test shell collections in 2 and 3 dimensions."""
    sc = spherical_droplet.ShellCollection.generate(dim=dim)

    for shell in sc:
        vs, ws = shell.vectors, shell.weights
        assert vs.shape[1] == dim
        assert vs.shape[0] == len(ws)
        assert ws.sum() == pytest.approx(1)
        np.testing.assert_allclose(ws @ vs, np.zeros(dim), atol=1e-10)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets(dim, rng):
    """Simple test of SphericalDropletAgents."""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(rng=rng), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == 1

    coupling = spherical_droplet.SphericalDropletActor()
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

    # test whether plotting works in principle
    if dim == 2:
        coupling.plot_shell_points(droplets, field)

    # test incompatible dimensions
    droplet_dim = (None, 2, 1, 1)[dim]
    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([1] * droplet_dim, 1)]
    )
    coupling = spherical_droplet.SphericalDropletActor()

    # can still make the evolver since it is mean-field model
    coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_const_shell_count(dim, rng):
    """Simple test of SphericalDropletAgents."""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(rng=rng), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == 1

    coupling = spherical_droplet.SphericalDropletActor(
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
    evolver((droplets._data_numba, field._data_numba), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    assert droplets.data[0].radius == radius

    droplets2 = droplets.copy(method="data")
    assert droplets2 is not droplets
    assert np.array_equal(droplets2.data, droplets.data)

    # test whether plotting works in principle
    if dim == 2:
        coupling.plot_shell_points(droplets, field)

    # test incompatible dimensions
    droplet_dim = (None, 2, 1, 1)[dim]
    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([1] * droplet_dim, 1)]
    )
    coupling = spherical_droplet.SphericalDropletActor()

    # can still make the evolver since it is mean-field model
    coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_reactions_inside(dim, compiled):
    """Simple test of SphericalDropletAgents with reactions."""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    d1 = SphericalDropletsElement.from_droplets([SphericalDroplet([1] * dim, 1)])
    c1 = spherical_droplet.SphericalDropletActor()

    d2 = SphericalDropletsElement.from_droplets([SphericalDroplet([2] * dim, 1)])
    c2 = spherical_droplet.SphericalDropletActor({"mean_reaction_inside": "-1"})

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


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_reactions_outside(dim, compiled):
    """Simple test of SphericalDropletAgents with reactions."""
    if compiled and dim == 2 and not module_available("numba_scipy"):
        pytest.skip("Python module `numba_scipy` not installed")

    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    d1 = SphericalDropletsElement.from_droplets([SphericalDroplet([1] * dim, 1)])
    c1 = spherical_droplet.SphericalDropletActor()

    d2 = SphericalDropletsElement.from_droplets([SphericalDroplet([2] * dim, 1)])
    c2 = spherical_droplet.SphericalDropletActor({"reaction_outside": "-1"})

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


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_material_conservation(backend):
    """Test whether the simulation conserves the total amount of material."""
    grid = UnitGrid([4] * 3, periodic=True)
    field_data = ScalarField(grid, 1.5)

    field = ScalarFieldElement.from_field(field_data)
    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([2] * 3, 0.5)], parameters={"droplet_concentration": 3}
    )
    state = State({"droplets": droplets, "field": field})
    total_amount = state.get_total_quantity("total_amount")

    coupling = spherical_droplet.SphericalDropletActor(
        {"equilibrium_concentration": "1"}
    )

    sim = Simulation(state)
    sim.add_actor(("droplets", "field"), coupling)
    res = sim.run(t_range=10, backend=backend)

    assert res.get_total_quantity("total_amount") == pytest.approx(total_amount)


@pytest.mark.skipif(
    not module_available("phasesep"), reason="requires `py-phasesep` module"
)
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_linear_reactions(backend):
    """Test whether the simulation with linear droplets conserves material."""
    cIn = 3
    cOut = "0.05 + 0.01/R"
    diff = 3
    k = 0.01
    c0 = 0.25
    reaction_flux = f"-{k} * (c - {c0})"

    # initialize some droplets
    droplets = SphericalDropletsElement.from_droplets(
        [SphericalDroplet([2] * 3, 0.5)], parameters={"droplet_concentration": cIn}
    )

    # initialize the background field
    grid = UnitGrid([4] * 3, periodic=True)
    field_data = ScalarField(grid, c0 - droplets.total_amount / grid.volume)
    field = ScalarFieldElement.from_field(field_data)

    # get the total amount
    state = State({"droplets": droplets, "background": field})
    total_amount = state.get_total_quantity("total_amount")
    assert total_amount == pytest.approx(c0 * grid.volume)

    simulation = Simulation(state)
    coupling = spherical_droplet.SphericalDropletActor(
        {
            "equilibrium_concentration": cOut,
            "diffusivity": diff,
            "reaction_outside": reaction_flux,
            "background_correction": True,
        }
    )
    simulation.add_actor(("droplets", "background"), coupling)

    simulation.add_actor(
        "background",
        ReactionDiffusionActor({"diffusivity": diff, "reaction_flux": reaction_flux}),
    )

    res = simulation.run(t_range=10, backend=backend)

    # check whether the reaction inside was determined correctly
    sIn = float(coupling.parameters["mean_reaction_inside"])
    assert sIn == pytest.approx(-k * (cIn - c0))
    total_amount = res.get_total_quantity("total_amount")
    assert total_amount == pytest.approx(total_amount, rel=1e-6)


@pytest.mark.skipif(
    not module_available("phasesep"), reason="requires `phasesep` module"
)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_linearized_fluxes(dim, rng):
    """A simple test for implementation of linearized fluxes for Active Emulsions under
    mean-field conditions."""
    if dim == 2 and not module_available("numba_scipy"):
        pytest.skip("Module `numba_scipy` not available")

    # make meanfield grid
    grid = CartesianGrid([(0, 1000)] * dim, 1, periodic=True)
    background = ScalarFieldElement.from_field(ScalarField(grid, 0.005))

    # Initialize radius and positions of 100 droplets
    droplet_data = [
        SphericalDroplet(
            position=grid.get_random_point(rng=rng), radius=rng.uniform(4, 6)
        )
        for _ in range(10)
    ]
    droplets = SphericalDropletsElement.from_droplets(droplet_data)
    state = State({"background": background, "droplets": droplets})

    # Initialize simulation
    reaction_flux = "0.001 - 0.01 * c"
    simulation = Simulation(state)
    simulation.add_actor(
        "background", ReactionDiffusionActor({"reaction_flux": reaction_flux})
    )
    droplet_actor = spherical_droplet.SphericalDropletActor(
        {"reaction_outside": reaction_flux}
    )
    simulation.add_actor(("droplets", "background"), droplet_actor)

    # Run simulation
    result = simulation.run(t_range=1000)

    # Check if droplet radii within the second decimal place
    final_droplet_radii = result["droplets"].data["radius"]
    np.testing.assert_allclose(
        final_droplet_radii, final_droplet_radii.mean(), rtol=0.1, atol=0
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_coarsening(dim, rng):
    """Simple test of coarsening."""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})

    emulsion = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(rng=rng), 0.1),
            SphericalDroplet(grid.get_random_point(rng=rng), 0.2),
        ]
    )
    droplets = SphericalDropletsElement.from_droplets(emulsion)

    coupling = spherical_droplet.SphericalDropletActor()

    ceq = coupling.get_equilibrium_concentrations(droplets).mean()
    field.concentration = ceq

    total_amount = pytest.approx(field.total_amount + droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.1)
    assert field.total_amount + droplets.total_amount == total_amount

    assert droplets.data[0]["radius"] < 0.1
    assert droplets.data[1]["radius"] > 0.2


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_drift(dim, backend):
    """Test drift direction of droplets."""
    grid = UnitGrid([4] * dim)
    # initialize gradient along x-direction
    field_data = ScalarField.from_expression(grid, "x / 40")

    for drift in [True, False]:
        field = ScalarFieldElement.from_field(field_data)
        droplets = SphericalDropletsElement.from_droplets(
            [SphericalDroplet([2] * dim, 0.2)]
        )
        state = State({"droplets": droplets, "field": field})

        coupling = spherical_droplet.SphericalDropletActor({"drift_enabled": drift})

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


def test_multithreading(rng):
    """Simple consistency test for multiprocessing."""
    grid = UnitGrid([1])
    field1 = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    field2 = field1.copy(method="data")

    emulsion = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(rng=rng), rng.uniform(0.01, 0.02))
            for _ in range(100)
        ]
    )
    droplets1 = SphericalDropletsElement.from_droplets(emulsion)
    droplets2 = SphericalDropletsElement.from_droplets(emulsion)

    coupling1 = spherical_droplet.SphericalDropletActor({"num_threads": 1})
    coupling2 = spherical_droplet.SphericalDropletActor({"num_threads": 2})

    evolver1 = coupling1.make_evolver_numba((droplets1, field1))
    evolver2 = coupling2.make_evolver_numba((droplets2, field2))

    evolver1((droplets1._data_numba, field1._data_numba), 0, 0.001)
    evolver2((droplets2._data_numba, field2._data_numba), 0, 0.001)

    assert_recarrays_allclose(droplets1.data, droplets2.data, atol=1e-5)
    np.testing.assert_allclose(field1.data, field2.data, rtol=0.1)
