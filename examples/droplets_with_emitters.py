#!/usr/bin/env python3

import sim
from droplets import SphericalDroplet
from pde import UnitGrid

# setup state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement(parameters={"grid": grid})
droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background, "droplets": droplets})

# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor("background", sim.DiffusionActor(parameters={"diffusivity": 0.1}))
positions = [grid.get_random_point() for _ in range(5)]
simulation.add_actor(
    "background", sim.EmittersActor({"positions": positions, "strengths": 10})
)
simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
