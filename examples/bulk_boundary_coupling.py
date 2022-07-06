#!/usr/bin/env python3
"""
Bulk-boundary coupling 
======================

Demonstrates coupling of bulk and boundary fields.
"""

import numpy as np

import pde

import sim

# set up a 2d cytosol with a 1d boundary membrane
grid = pde.UnitGrid([32, 8], periodic=[True, False])
cytosol = sim.ScalarFieldElement.from_field(pde.ScalarField(grid, 0.001))
membrane = sim.ScalarBoundaryFieldElement.from_domain(
    cytosol, axis=1, upper=True, data=np.random.uniform(-1, 1, 32)
)
state = sim.State({"cytosol": cytosol, "membrane": membrane})

# set up simulation of phase separation in membrane and diffusion in cytosol
simulation = sim.Simulation(state)
simulation.add_actor("cytosol", sim.DiffusionActor())
simulation.add_actor("membrane", sim.ScalarPDEActor(pde.CahnHilliardPDE()))
# couple both domains using an exchange flux
boundary_coupling = sim.FieldBoundaryCouplingActor(
    {"exchange_flux": "0.1 * (bulk - boundary)"}
)
simulation.add_actor(("cytosol", "membrane"), boundary_coupling)

# run the simulation
result = simulation.run(t_range=100)
result.plot()
