TODO
====
* Allow adding (periodic) boundaries for Brownian motion
* Add option of creating actor from factory function.
    - The function would receive the elements and return an evolver
    - This factor function could be directly used in `add_actor` 
* Improve plotting
    - adjust color of droplets
    - adjust bounds for elements
* Convenient I/O for states and simulations (actor + couplings)
* Add napari plotting
* Allow easy addition of the simulation parameters when writing the state
  to a file (for documentation purposes)
* Simulation:
	Add diagnostic information (dt, step_count, degrees of freedom)
 