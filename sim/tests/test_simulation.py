"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from droplets import SphericalDroplet
from pde import DiffusionPDE, ScalarField, UnitGrid
from pde.tools.misc import module_available

from .. import *


def test_simulation():
    """test some methods of the Simulation class"""

    # setup state
    grid = UnitGrid([32, 32], periodic=True)
    background = ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
    droplets = SphericalDropletsElement.from_droplets(droplet_data)
    state = State({"background": background, "droplets": droplets})

    # setup simulation
    simulation = Simulation(state, actors=[("background", DiffusionActor())])
    simulation.add_actor(("droplets", "background"), SphericalDropletActor())

    with pytest.raises(ValueError):
        simulation.add_actor("nonsense", DiffusionActor())
    with pytest.raises(ValueError):
        simulation.add_actor(("background", "background"), DiffusionActor())
    with pytest.raises(RuntimeError):
        simulation.add_actor(("droplets",), DiffusionActor(), check="raise")
    with pytest.raises(RuntimeError):
        simulation.add_actor(
            ("droplets", "background"), SphericalDropletActor(), check="raise"
        )

    assert isinstance(str(simulation), str)
    assert isinstance(repr(simulation), str)
    assert isinstance(simulation.info, dict)
    assert len(simulation.info["actors"]) == 2

    if module_available("networkx"):
        simulation.plot_as_graph()
        simulation.plot_interacting_elements()

    # run simulation
    simulation.run(t_range=10)


def test_simulation_timing():
    """test some methods of the Simulation class"""
    # set up state
    field = ScalarField.random_uniform(UnitGrid([8, 8], periodic=True))
    element = ScalarFieldElement.from_field(field)
    state = State({"field": element})

    # set up simulation
    simulation = Simulation(state, profile=True)
    eq = DiffusionPDE(diffusivity=0.1)
    simulation.add_actor("field", ScalarPDEActor(eq))

    # run simulation using the numpy backend
    simulation.run(t_range=1, backend="numpy")
    timings = simulation.timings
    assert len(timings) == 1
    assert timings[0] > 0

    # run simulation using the numba backend
    simulation.run(t_range=1, backend="numba")
    timings = simulation.timings
    assert len(timings) == 1
    assert timings[0] > 0
