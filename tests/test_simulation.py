"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import SphericalDroplet
from pde import CartesianGrid, DiffusionPDE, ScalarField, UnitGrid
from pde.tools.misc import module_available, skipUnlessModule

import sim
from helpers import assert_recarrays_allclose


def test_simulation():
    """test some methods of the Simulation class"""
    # setup state
    grid = UnitGrid([32, 32], periodic=True)
    background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
    droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
    state = sim.State({"background": background, "droplets": droplets})

    # setup simulation
    simulation = sim.Simulation(state, actors=[("background", sim.DiffusionActor())])
    simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor())

    with pytest.raises(ValueError):
        simulation.add_actor("nonsense", sim.DiffusionActor())
    with pytest.raises(ValueError):
        simulation.add_actor(("background", "background"), sim.DiffusionActor())
    with pytest.raises(TypeError):
        simulation.add_actor(("droplets",), sim.DiffusionActor(), check="raise")
    with pytest.raises(RuntimeError):
        simulation.add_actor(
            ("droplets", "background"), sim.SphericalDropletActor(), check="raise"
        )

    assert isinstance(str(simulation), str)
    assert isinstance(repr(simulation), str)
    assert isinstance(simulation.info, dict)
    assert len(simulation.info["actors"]) == 2

    if module_available("networkx"):
        simulation.plot_as_graph()
        simulation.plot_interacting_elements()

    # run simulation
    simulation.run(t_range=10, backend="numpy", tracker=None)
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) < 5
    simulation.run(t_range=10, backend="numba", tracker=None)
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) < 40


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_adaptive_simulation_simple(backend):
    """test some adaptive simulations"""
    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True))
    element = sim.ScalarFieldElement.from_field(field)
    state = sim.State({"field": element})

    # prepare simulation
    simulation = sim.Simulation(state.copy(method="data"))
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", sim.ScalarPDEActor(eq))
    simulation2 = simulation.copy(method="data")

    # run simulation using fixed and adaptive time steps
    simulation.run(t_range=1, backend=backend, tracker=None)
    result = simulation.state["field"].data
    simulation2.run(t_range=1, backend=backend, adaptive=True, tracker=None)
    np.testing.assert_allclose(result, simulation2.state["field"].data)
    thresh = {"numpy": 5, "numba": 30}[backend]
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) < thresh


@skipUnlessModule("phasesep")
def test_adaptive_reaction_diffusion():
    """test adaptive reaction-diffusion simulation"""
    # set up state
    grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)
    field = ScalarField.random_uniform(grid)
    element = sim.ScalarFieldElement.from_field(field)
    state = sim.State({"field": element})

    # prepare simulation
    simulation = sim.Simulation(state)
    simulation.add_actor(
        "field", sim.ReactionDiffusionActor({"reaction_flux": "0.01 - 0.1 * c"})
    )
    result = simulation.run(t_range=100, adaptive=True, tracker=None)

    np.testing.assert_allclose(result["field"].data, 0.1, atol=1e-4)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_adaptive_simulation_complex(backend):
    """test some adaptive simulations"""
    thresh = {"numpy": 5, "numba": 25}[backend]

    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True))
    element = sim.ScalarFieldElement.from_field(field)
    state = sim.State({"field": element})

    # run simulation using fixed time steps
    simulation = sim.Simulation(state.copy(method="data"))
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", sim.ScalarPDEActor(eq))
    simulation.run(t_range=1, backend=backend, tracker=None)
    result = simulation.state["field"].data
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) < thresh

    # run simulation using adaptive time steps
    simulation = sim.Simulation(state.copy(method="data"))
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", sim.ScalarPDEActor(eq))
    simulation.run(t_range=1, backend=backend, adaptive=True, tracker=None)
    np.testing.assert_allclose(result, simulation.state["field"].data)
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) < thresh


def test_simulation_timing():
    """test some methods of the Simulation class"""
    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True))
    element = sim.ScalarFieldElement.from_field(field)
    state = sim.State({"field": element})

    # set up simulation
    simulation = sim.Simulation(state, profile=True)
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", sim.ScalarPDEActor(eq))

    # run simulation using the numpy backend
    simulation.run(t_range=1, backend="numpy", tracker=None)
    timings = simulation.timings
    assert len(timings) == 1
    assert timings[0] > 0

    # run simulation using the numba backend
    simulation.run(t_range=1, backend="numba", tracker=None)
    timings = simulation.timings
    assert len(timings) == 1
    assert timings[0] > 0


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_simulation_cache(backend, use_cache):
    """test caching of Simulation class"""
    # setup state
    grid = UnitGrid([4, 4], periodic=True)
    background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
    droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
    state = sim.State({"background": background, "droplets": droplets})
    state2 = state.copy(method="clean")

    # setup simulation
    diff_actor = sim.DiffusionActor()
    simulation = sim.Simulation(state, actors=[("background", diff_actor)])
    simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor())

    # run simulation
    jit_max = {"numba": 40, "numpy": 5}[backend]
    simulation.run(t_range=10, backend=backend, tracker=None, use_cache=True)
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) <= jit_max

    simulation.state = state2
    simulation.run(t_range=10, backend=backend, tracker=None, use_cache=use_cache)
    if use_cache and backend == "numba":
        jit_max = 0
    assert sum(simulation.diagnostics["controller"]["jit_count"].values()) <= jit_max

    assert_recarrays_allclose(state["droplets"].data, state2["droplets"].data)
    np.testing.assert_allclose(state["background"].data, state2["background"].data)
