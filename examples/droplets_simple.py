#!/usr/bin/env python3

from pde import UnitGrid, ScalarField
from droplets import SphericalDroplet
import sim

# setup state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({'background': background, 'droplets': droplets})

# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor('background', sim.DiffusionActor())
simulation.add_actor(('droplets', 'background'), sim.SphericalDropletActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
