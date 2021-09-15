'''
.. codeauthor:: Ajinkya Kulkarni <ajinkya.kulkarni@ds.mpg.de>
'''

import pytest
import numpy as np
from pde.tools.misc import skipUnlessModule
from pde import CartesianGrid, ScalarField
from droplets import SphericalDroplet

import sys

sys.path.append('../../../../../py-sim')
sys.path.append('../../../../../py-phasesep')

from .... import ScalarFieldElement, SphericalDropletsElement, State, Simulation, ReactionDiffusionActor, SphericalDropletActor

# @skipUnlessModule("phasesep")
@pytest.mark.parametrize("backend", ['numpy', 'numba'])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_linearized_fluxes(dim, backend):
    """a Simple test for implementation of linearized fluxes for Active Emulsions under mean-field conditions"""
    # Make meanfield grid
    grid = CartesianGrid([(0, 1000)] * dim, 1, periodic=True)
    background = ScalarFieldElement.from_field(ScalarField(grid, 0.005))

    # Initialize radius and positions of 100 droplets
    droplet_data = [SphericalDroplet(position=grid.get_random_point(), radius=np.random.uniform(4, 6))
                        for _ in range(100)]
    droplets = SphericalDropletsElement.from_droplets(droplet_data)
    state = State({"background": background, "droplets": droplets})

    # Initialize simulation
    reaction_flux = "0.001 - 0.01 * c"
    simulation = Simulation(state)
    simulation.add_actor("background", ReactionDiffusionActor({"reaction_flux": reaction_flux}))
    droplet_actor = SphericalDropletActor({'shell_thickness': grid.discretization[0],
                                               'shell_sector_size': grid.discretization[0],
                                               'reaction_inside': -0.01,
                                               'reaction_outside': reaction_flux})
    simulation.add_actor(("droplets", "background"), droplet_actor)

    # Run simulation
    result = simulation.run(t_range=int(1e3), backend='numba')

    # Check if droplet radii within the second decimal place
    final_droplet_radii = result["droplets"].data["radius"]
    np.testing.assert_allclose(final_droplet_radii, final_droplet_radii.mean(), rtol=0.1, atol=0)
