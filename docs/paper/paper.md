---
title: 'emulsim: A Python package for simulating complex emulsions'
tags:
  - Python
  - physical simulation
  - phase separation
  - biomolecular condensates
authors:
  - name: David Zwicker
    orcid: 0000-0002-3909-3334
    affiliation: 1
affiliations:
 - name: Max Planck Institute for Dynamics and Self-Organization, Göttingen, Germany
   index: 1
date: June 2026
bibliography: paper.bib
---

# Summary

Abstract

# Statement of need

Emulsions in complex situations with reactions [cite our work and christoph, and some more]
Mention interactions in gradients [Eric's work]

State-of-the art
* Current approach: MD or other particle based simulations -> too complex
* Field-based approach, i.e., based on py-pde
* Custom code, which can be hard to develop

We have developed a general method in [@Kulkarni2023], but the code is not easily accessible and extensible.


# Methods

The `emulsim` package provides a general simulation framework, where the state of the system is described by various *elements*.
This state then evolves in times using various *actors*, which affect one or many elements.
Building on this general principle, the package particularly provides elements that are useful to described emulsions and their surroundings.
For instance, we provide the basic class `SphericalDropletsElement`, which describes a collection of droplets by their positions and sizes in one to three dimensions.
We also provide a class `MulticomponentDroplet`, which additionally allows specifying the composition of droplets.
Finally, we provide various elements to describe the surroundings of droplets, most notably the `MeanfieldElement`, which assumes a homogeneous composition, and `ScalarFieldElement`, where a scalar field captures the background composition.
All element classes provide general methods for storing, reading, and plotting data to enable an interactive workflow based on jupyter notebooks.
The state of the system is described by an instance of the `State` class, which may contain multiple different elements.

In the `emulsim` package, the dynamics are defined by a `Simulation` object, which contains one or more actors that each affect one or multiple elements.
This approach allows combining multiple actors without redefining their code simply by combining them in a `Simulation`.
Each actor inherits from `ActorBase`, which defines the necessary behavior.
In the simplest case, a custom actor only needs to overwrite the evolve() method, which evolves its elements from time t to t + dt, changing the respective data attributes in place:

Autonomous actors

Coupling actors

Simulation approach

Simulations are implemented in a pure Python implementation and a version accelerated by just-in-time compilation using `numba` [@lam2015Numba]

# Examples

The following code illustrates the main functionality of the package by simulating a passive set of droplets in a common background.
Maybe we can adjust the simulation to give the droplet size distribution in the end?

```python
import pandas as pd
import matplotlib.pyplot as plt
from droplets import Emulsion  # `py-droplets` package for initializing droplets
from pde import CartesianGrid, ScalarField  # `py-pde` for initializing fields
import sim  # `py-sim` package discussed here
```

```python
# set up initial state
grid = CartesianGrid([[0, 1000], [0, 1000]], 128, periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
# initialize 1000 droplets with uniformly distributed radii between 1 and 10
droplet_data = Emulsion.from_random(1000, grid, radius=(1, 10))
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State(
    {"background": background, "droplets": droplets},
    parameters={"bounds": grid.axes_bounds},
)
```

```python
# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor("background", sim.DiffusionActor())
droplet_actor = sim.SphericalDropletActor({"equilibrium_concentration": "1 / radius"})
simulation.add_actor(("droplets", "background"), droplet_actor)
```

```python
tracker = sim.DropletElementTracker("droplets", interrupts=1000)  # record data
result = simulation.run(t_range=50_000, tracker=["progress", tracker])  # run simulation
```

```python
result.plot(element_args={"droplets": {"set_bounds": True, "grid": grid}})  # last state
tracker.droplet_tracks.plot()  # droplet radii as a function of time
```

![Final state of the simulation](pd.jpg "Phase diagram of binary mixture")


# Acknowledgements

I thank all current and past members of the Zwicker group for stimulating discussions.
We gratefully acknowledge funding from the Max Planck Society and the European Union (ERC, EmulSim, 101044662).

# References
