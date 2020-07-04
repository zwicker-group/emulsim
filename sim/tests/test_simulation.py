"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from droplets import SphericalDroplet
from pde import UnitGrid, ScalarField
from pde.tools.misc import module_available

from .. import *


def test_simulation():
    """ test some methods of the Simulation class """

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

    assert isinstance(str(simulation), str)
    assert isinstance(repr(simulation), str)
    assert isinstance(simulation.info, dict)
    assert len(simulation.info["actors"]) == 2

    if module_available("networkx"):
        simulation.plot_as_graph()
        simulation.plot_interacting_elements()

    # run simulation
    simulation.run(t_range=10)
