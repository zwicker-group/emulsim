"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest

from droplets import SphericalDroplet
from pde import CartesianGrid, DiffusionPDE, ScalarField, UnitGrid
from pde.backends.numba.utils import JIT_COUNT
from pde.tools.misc import module_available

import emulsim
from helpers import assert_recarrays_allclose


def test_simulation(rng):
    """Test some methods of the Simulation class."""
    # setup state
    grid = UnitGrid([32, 32], periodic=True)
    background = emulsim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [
        SphericalDroplet(grid.get_random_point(rng=rng), 1) for _ in range(3)
    ]
    droplets = emulsim.SphericalDropletsElement.from_droplets(droplet_data)
    state = emulsim.State({"background": background, "droplets": droplets})

    # setup simulation
    simulation = emulsim.Simulation(
        state, actors=[("background", emulsim.DiffusionActor())]
    )
    simulation.add_actor(("droplets", "background"), emulsim.SphericalDropletActor())

    with pytest.raises(ValueError):
        simulation.add_actor("nonsense", emulsim.DiffusionActor())
    with pytest.raises(ValueError):
        simulation.add_actor(("background", "background"), emulsim.DiffusionActor())
    with pytest.raises(TypeError):
        simulation.add_actor(("droplets",), emulsim.DiffusionActor(), check="raise")
    with pytest.raises(RuntimeError):
        simulation.add_actor(
            ("droplets", "background"), emulsim.SphericalDropletActor(), check="raise"
        )

    assert isinstance(str(simulation), str)
    assert isinstance(repr(simulation), str)
    assert isinstance(simulation.info, dict)
    assert len(simulation.info["actors"]) == 2

    if module_available("networkx"):
        simulation.plot_as_graph()
        simulation.plot_interacting_elements()

    # run simulation
    jit_count = int(JIT_COUNT)
    simulation.run(t_range=10, backend="numpy", tracker=None)
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count < 5

    jit_count = int(JIT_COUNT)
    simulation.run(t_range=10, backend="numba", tracker=None)
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count < 50


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_adaptive_simulation_simple(backend, rng):
    """Test some adaptive simulations."""
    jit_count = int(JIT_COUNT)

    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True), rng=rng)
    element = emulsim.ScalarFieldElement.from_field(field)
    state = emulsim.State({"field": element})

    # prepare simulation
    simulation = emulsim.Simulation(state.copy(method="data"))
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", emulsim.ScalarPDEActor(eq))
    simulation2 = simulation.copy(method="data")

    # run simulation using fixed and adaptive time steps
    simulation.run(t_range=1, backend=backend, tracker=None)
    result = simulation.state["field"].data
    simulation2.run(t_range=1, backend=backend, adaptive=True, tracker=None)
    np.testing.assert_allclose(result, simulation2.state["field"].data)
    thresh = {"numpy": 5, "numba": 30}[backend]
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count < 40


def test_adaptive_reaction_diffusion(rng):
    """Test adaptive reaction-diffusion simulation."""
    # set up state
    grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)
    field = ScalarField.random_uniform(grid, rng=rng)
    element = emulsim.ScalarFieldElement.from_field(field)
    state = emulsim.State({"field": element})

    # prepare simulation
    simulation = emulsim.Simulation(state)
    simulation.add_actor(
        "field", emulsim.ReactionDiffusionActor({"reaction_flux": "0.01 - 0.1 * c"})
    )
    result = simulation.run(t_range=100, adaptive=True, tracker=None)

    np.testing.assert_allclose(result["field"].data, 0.1, atol=1e-4)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_adaptive_simulation_complex(backend, rng):
    """Test some adaptive simulations."""
    thresh = {"numpy": 5, "numba": 25}[backend]

    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True), rng=rng)
    element = emulsim.ScalarFieldElement.from_field(field)
    state = emulsim.State({"field": element})

    # run simulation using fixed time steps
    jit_count = int(JIT_COUNT)
    simulation = emulsim.Simulation(state.copy(method="data"))
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", emulsim.ScalarPDEActor(eq))
    simulation.run(t_range=1, backend=backend, tracker=None)
    result = simulation.state["field"].data
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count < thresh

    # run simulation using adaptive time steps
    jit_count = int(JIT_COUNT)
    simulation = emulsim.Simulation(state.copy(method="data"))
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", emulsim.ScalarPDEActor(eq))
    simulation.run(t_range=1, backend=backend, adaptive=True, tracker=None)
    np.testing.assert_allclose(result, simulation.state["field"].data)
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count < thresh


def test_simulation_timing(rng):
    """Test some methods of the Simulation class."""
    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True), rng=rng)
    element = emulsim.ScalarFieldElement.from_field(field)
    state = emulsim.State({"field": element})

    # set up simulation
    simulation = emulsim.Simulation(state, profile=True)
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", emulsim.ScalarPDEActor(eq))

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
def test_simulation_cache(backend, use_cache, rng):
    """Test caching of Simulation class."""
    # setup state
    grid = UnitGrid([4, 4], periodic=True)
    background = emulsim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [
        SphericalDroplet(grid.get_random_point(rng=rng), 1) for _ in range(3)
    ]
    droplets = emulsim.SphericalDropletsElement.from_droplets(droplet_data)
    state = emulsim.State({"background": background, "droplets": droplets})
    state2 = state.copy(method="clean")

    # setup simulation
    diff_actor = emulsim.DiffusionActor()
    simulation = emulsim.Simulation(state, actors=[("background", diff_actor)])
    simulation.add_actor(("droplets", "background"), emulsim.SphericalDropletActor())

    # run simulation
    jit_max = {"numba": 40, "numpy": 5}[backend]
    jit_count = int(JIT_COUNT)
    simulation.run(t_range=10, backend=backend, tracker=None, use_cache=True)
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count <= jit_max

    simulation.state = state2
    jit_count = int(JIT_COUNT)
    simulation.run(t_range=10, backend=backend, tracker=None, use_cache=use_cache)
    if use_cache and backend == "numba":
        jit_max = 0
    if not nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) - jit_count <= jit_max

    assert_recarrays_allclose(state["droplets"].data, state2["droplets"].data)
    np.testing.assert_allclose(state["background"].data, state2["background"].data)
