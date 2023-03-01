"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import SphericalDroplet
from pde import DiffusionPDE, ScalarField, UnitGrid
from pde.tools.misc import module_available

import sim


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
    simulation.run(t_range=10, tracker=None)


def test_simulation_values():
    """test some methods of the Simulation class"""
    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True))
    element = sim.ScalarFieldElement.from_field(field)
    state = sim.State({"field": element})

    # set up simulation
    simulation = sim.Simulation(state.copy())
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", sim.ScalarPDEActor(eq))

    # run simulation using the numba backend
    simulation.run(t_range=1, backend="numba", tracker=None)
    result = simulation.state["field"].data

    for adaptive in [True, False]:
        # run simulation using the numpy backend
        simulation = sim.Simulation(state.copy())
        eq = DiffusionPDE(diffusivity=0.1)
        simulation.add_actor("field", sim.ScalarPDEActor(eq))
        simulation.run(t_range=1, backend="numpy", adaptive=adaptive, tracker=None)
        np.testing.assert_allclose(result, simulation.state["field"].data)


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
