Examples
^^^^^^^^

We here collect examples for using the package to demonstrate some of its
functionality. 


Basic simulation
""""""""""""""""

Basic simulations can be run as follows

.. include:: ../examples/simple.rst

Here, we define a list of :class:`~droplets.droplets.SphericalDroplet` an use it
to initialize
:class:`~agent_based.agents.spherical_droplet.SphericalDropletAgents`.
Together with a background
:class:`~agent_based.backgrounds.scalar_field.DiffusionBackground` that
simulates simple diffusion, we then create the full
:class:`~agent_based.state.State` of the simulation.
Finally, :func:`~agent_based.solver.simulate_agents` is used to run the
simulation.
The `result` is an instance of :class:`~agent_based.state.State` storing the
state of the system at the final time point.


Custom agent class
""""""""""""""""""

One strength of the package is that agents can be simply defined, as shown below

.. include:: ../examples/agents_custom.rst

Here, we define agents that behave as Brownian particle, i.e., they simply
diffuse around and do not interact with each other or the background. This
example can be used as the basis to augmented the agents, e.g., with fluxes
between the agents and the background.

Custom background class
"""""""""""""""""""""""

Similarly, the behavior of the background can be changed:

.. include:: ../examples/background_custom_class.rst

In this example, the background dynamics are governed by a Cahn-Hilliard
equation instead of the simple diffusion equation.
