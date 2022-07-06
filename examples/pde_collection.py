#!/usr/bin/env python3
"""
Multiple coupled PDEs
=====================

Demonstrates how multiple PDEs can be coupled.
"""


from pde import PDE, FieldCollection, ScalarField, UnitGrid

import sim

# parameters
a, b = 1, 3
d0, d1 = 1, 0.1

# set up state
grid = UnitGrid([64, 64])
u = ScalarField(grid, a, label="Field $u$")
v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
field = FieldCollection([u, v])
element = sim.FieldCollectionElement.from_field(field)
state = sim.State({"field": element})

# set up simulation
simulation = sim.Simulation(state)
eq = PDE(  # Brusselator equations
    {
        "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
        "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
    }
)
simulation.add_actor("field", sim.CollectionPDEActor(eq))

# run simulation
result = simulation.run(t_range=20, dt=1e-3)

result.plot()
