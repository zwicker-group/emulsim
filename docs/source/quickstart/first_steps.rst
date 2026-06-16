First steps
^^^^^^^^^^^

We here collect simple examples for using the package to demonstrate some of its
functionality. 


Basic PDE simulation
""""""""""""""""""""

A simple simulation of a partial differential equation (PDE) can be run as follows.

.. include:: ../examples/pde_simple.rst

General, each simulation is split in three parts: First, the simulation elements
are combined to a simulation state. This defines *what* can be evolved in time.
Second, the actors are added to a simulation. This determines *how* the elements
change in time. Finally, the simulation is actually run and results are collected. 
Here, `result` is a :class:`~emulsim.state.State` class like `state`, but with
updated data.


Droplets simulation
"""""""""""""""""""

The package becomes useful, when multiple elements interact. A simple example
are droplets exchange material via a background phase:

.. include:: ../examples/droplets_simple.rst


Custom actor
""""""""""""

One strength of the package is that actors can be simply defined, as shown below

.. include:: ../examples/custom_brownian_particles.rst

Here, we define an actor that moves points like Brownian particle, i.e., they
simply diffuse around and do not interact with each other.

