#!/usr/bin/env python3

from pde import UnitGrid, ScalarField, DiffusionPDE
import sim

# setup state
field = ScalarField.random_uniform(UnitGrid([32, 32], periodic=True))
element = sim.ScalarFieldElement.from_field(field)
state = sim.State({'field': element})

# setup simulation
simulation = sim.Simulation(state)
eq = DiffusionPDE(diffusivity=0.1)
simulation.add_actor('field', sim.ScalarPDEActor(eq))

# run simulation
result = simulation.run(t_range=10)

result.plot()
