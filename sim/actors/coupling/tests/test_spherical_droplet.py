"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
.. codeauthor:: Ajinkya Kulkarni <ajinkya.kulkarni@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import spatial

from droplets import Emulsion, SphericalDroplet
from pde import CartesianGrid, ScalarField, UnitGrid
from pde.grids.base import DimensionError
from pde.tools.misc import skipUnlessModule

from .... import ReactionDiffusionActor, Simulation, State
from ....elements import MeanfieldElement, ScalarFieldElement, SphericalDropletsElement
from .. import spherical_droplet


def recarrays_allclose(a, b):
    """tests whether the entries of two structured arrays are all close"""
    if a.dtype != b.dtype:
        return False
    return all(np.allclose(a[name], b[name]) for name in a.dtype.names)


def test_spherical_polygon_area():
    """test the function get_spherical_polygon_area"""
    area = spherical_droplet.get_spherical_polygon_area
    sector_area = area([[0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    assert sector_area == pytest.approx(np.pi / 2)


def test_spherical_voronoi():
    """test spatial.SphericalVoronoi"""
    # random points on the sphere
    ps = np.random.random((32, 3)) - 0.5
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
    """test spatial.SphericalVoronoi"""
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


def test_points_on_sphere_2():
    """special tests for 2 dimensions"""
    num = np.random.randint(3, 9)
    shell = spherical_droplet.PointsOnSphere.make_uniform(dim=2, num_points=num)
    assert num * shell.get_mean_separation() == pytest.approx(2 * np.pi)


def test_shells_1d():
    """test shell collection in 1 dimensions"""
    sc = spherical_droplet.ShellCollection.generate(dim=1)
    assert len(sc) == 1
    shell = sc.get_shell(1e3)
    assert shell.vectors.shape == (2, 1)
    assert shell.weights.shape == (2,)
    np.testing.assert_allclose(shell.weights, np.full(2, 0.5))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_shells_general(dim):
    """test shell collections in 2 and 3 dimensions"""
    sc = spherical_droplet.ShellCollection.generate(dim=dim)

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
    coupling = spherical_droplet.SphericalDropletActor()
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
    coupling = spherical_droplet.SphericalDropletActor()
    with pytest.raises(DimensionError):
        coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spherical_droplets_reactions_inside(dim, compiled):
    """simple test of SphericalDropletAgents with reactions"""
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
        evolver(state.data, 0, 0.5)
    else:
        sim.evolve(state, 0, 0.5)
    assert d1.total_amount > d2.total_amount


@skipUnlessModule("phasesep")
@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("dim", [3])  # 1, 2, 3])
def test_linearized_fluxes(dim, backend):
    """a simple test for implementation of linearized fluxes for Active Emulsions under
    mean-field conditions"""
    # make meanfield grid
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
    droplet_actor = spherical_droplet.SphericalDropletActor(
        {"mean_reaction_inside": -0.01, "reaction_outside": reaction_flux}
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

    coupling = spherical_droplet.SphericalDropletActor()

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

    coupling1 = spherical_droplet.SphericalDropletActor({"num_threads": 1})
    coupling2 = spherical_droplet.SphericalDropletActor({"num_threads": 2})

    evolver1 = coupling1.make_evolver_numba((droplets1, field1))
    evolver2 = coupling2.make_evolver_numba((droplets2, field2))

    evolver1((droplets1.data, field1.data), 0, 0.001)
    evolver2((droplets2.data, field2.data), 0, 0.001)

    np.testing.assert_allclose(field1.data, field2.data, rtol=0.1)
