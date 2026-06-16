#!/usr/bin/env python3
"""
Simple PDE
==========

Demonstrates a minimal example involving the diffusion equation
"""

from pde import DiffusionPDE, ScalarField, UnitGrid

import emulsim

# set up state
field = ScalarField.random_uniform(UnitGrid([32, 32], periodic=True))
element = emulsim.ScalarFieldElement.from_field(field)
state = emulsim.State({"field": element})

# set up simulation
simulation = emulsim.Simulation(state)
eq = DiffusionPDE(diffusivity=0.1)
simulation.add_actor("field", emulsim.ScalarPDEActor(eq))

# run simulation
result = simulation.run(t_range=10, dt=0.1)

result.plot()
