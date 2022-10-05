#!/usr/bin/env python3
r"""
Simulation of multicomponent droplets in 3D
===========================================

This example shows how to simulate droplets comprising two components.
"""

import numpy as np

from pde import CartesianGrid, FieldCollection

import sim

# set up state
grid = CartesianGrid([[0, 128]] * 3, 1)
fc = FieldCollection.scalar_random_uniform(2, grid, 0.01, 0.05)
background_el = sim.FieldCollectionElement.from_fields(fc)
droplet_data = [
    sim.MulticomponentDroplet.from_composition(
        position=grid.get_random_point(),
        radius=np.random.uniform(1, 5),
        phis=[0.4, 0.4],
    )
    for _ in range(20)
]
droplets_el = sim.MulticomponentDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background_el, "droplets": droplets_el})

# set up simulation
simulation = sim.Simulation(state)
droplets_actor = sim.MulticomponentDropletActor.from_linear_reactions(
    {
        "chis": [[-1, 0], [0, -1]],  # interactions between internal components
        "chis_solvent": [2.5, 2.5],  # interactions of components with solvent
    },
    rates=np.diag([-0.01, -0.01]),
    production=[0.002, 0.002],
)
simulation.add_actor(("droplets", "background"), droplets_actor)


# run simulation
result = simulation.run(t_range=1000, dt=1e-2)

result.plot(title=f"{result['droplets'].droplet_count} droplets")
