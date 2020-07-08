#!/usr/bin/env python3

import sim
from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid

# setup state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.005))
droplet_data = [SphericalDroplet(grid.get_random_point(), 0.1) for _ in range(10)]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background, "droplets": droplets})

# setup simulation
reaction_flux = "0.001 - 0.01 * c"
simulation = sim.Simulation(state)
simulation.add_actor(
    "background", sim.ReactionDiffusionActor({"reaction_flux": reaction_flux})
)
droplet_actor = sim.SphericalDropletActor(
    {"reaction_inside": -0.01, "reaction_outside": reaction_flux}
)
simulation.add_actor(("droplets", "background"), droplet_actor)

# run simulation
result = simulation.run(t_range=10)

result.plot()
