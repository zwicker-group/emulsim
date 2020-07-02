"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from droplets import SphericalDroplet

from pde import UnitGrid, ScalarField

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

    assert isinstance(str(simulation), str)
    assert isinstance(repr(simulation), str)
    assert isinstance(simulation.info, dict)
    assert len(simulation.info["actors"]) == 2

    # run simulation
    result = simulation.run(t_range=10)
