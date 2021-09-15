'''
.. codeauthor:: Ajinkya Kulkarni <ajinkya.kulkarni@ds.mpg.de>
'''

# A Simple test for implementation of linearized fluxes for Active Emulsions under mean-field conditions

import pytest
import numpy as np
from pde.tools.misc import skipUnlessModule
from pde import CartesianGrid, ScalarField
from droplets import SphericalDroplet
from ... import ReactionDiffusionBackground, SphericalDropletAgents, AgentSimulation, DropletAgentTracker

# import sys
# sys.path.append('../../../../py-phasesep')
@skipUnlessModule("phasesep")

@pytest.mark.parametrize("backend", ['numpy', 'numba'])
@pytest.mark.parametrize("dim", [1, 2, 3])

def test_linearized_fluxes(dim, backend):

    # Make grid

    grid = CartesianGrid([(0, int(1e3))]*dim, 1, periodic = True)

    # Initialize radius and positions of 100 droplets

    radius_of_droplets = np.random.uniform(4, 6, 100)

    position_of_droplets = []

    for j in range(100):

        position_of_droplets.append(grid.get_random_point())

    list_of_droplets = [SphericalDroplet(position = position_of_droplets[j], radius = radius_of_droplets[j])

                        for j in range(100)]

    # Initialize background and agents

    background = ReactionDiffusionBackground({'reaction_flux': '0.001 - 0.01 * c'})

    # Mean field simulations by setting shell_thickness and shell_sector_size to domain size

    agents = SphericalDropletAgents({'shell_thickness': grid.discretization[0],
    'shell_sector_size': grid.discretization[0],
    'reaction_inside': -0.01,
    'reaction_outside': '0.001 - 0.01 * c'})

     # Initialize simulation

    simulation = AgentSimulation(background, agents)

    background_scalarfield = ScalarField(grid, 0.005)

    background_plus_agents = simulation.get_state(background = background_scalarfield, agents = list_of_droplets)

    # Initialize Trackers

    droplet_tracker = DropletAgentTracker()

    # Run simulation

    result = simulation.run(background_plus_agents, t_range = int(1e3),
                            tracker = [droplet_tracker],
                            backend = backend)

    # Check if droplet radii within the second decimal place

    final_droplet_radii = droplet_tracker.emulsions.emulsions[-1].data['radius']

    np.testing.assert_allclose(final_droplet_radii, final_droplet_radii.mean(), rtol = 0.1, atol = 0)
