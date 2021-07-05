TODO
====
* Support random number generators
* Add interactive plotting using napari
    - add interactive tracker (need to update droplet radius, too!)
* Trackers should probably be defined with elements, since they track elements
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
* Think about introducing data class that holds integrated, global variables
    - this might be helpful to implement lagrange multipliers and the like
    - generally, we should use a state class that contains the state of a pde
      (in most cases, this would be a FieldBase)
    - the state class should also handle serialization and io with hdf