TODO
====
* Support FieldCollections as elements and actors
	- do we need a separate element?
	- we should be able to apply PDEs directly to the collection
	- how can we decide to which field other actors couple?
	  - one option would be to use braket notation, e.g.,
	    simulation.add_actor(['fieldA[0]', 'fieldB[1]'], actor)
	  the simulation then has to make sure that the correct sub-data is passed to the actor 
* Support storage for storing trajectories to memory and file
	- introduce the concept of trajectories of states
	- this is simple for MemoryStorage (could reuse py-pde?)
	- think about how to abstract this for FileStorage
	- this might depend on the py-model package
* Support random number generators
* Add test for re-running a simulation with use_cache
* Add interactive plotting using napari
    - add interactive tracker (need to update droplet radius, too!)
* Trackers should probably be defined with elements, since they track elements
* Allow adding (periodic) boundaries for Brownian motion
* Improve plotting
    - adjust color of droplets
    - adjust bounds for elements
* Convenient I/O for states and simulations (actor + couplings)
* Allow easy addition of the simulation parameters when writing the state
  to a file (for documentation purposes)
* Simulation:
	Add diagnostic information (dt, step_count, degrees of freedom)
* Think about introducing data class that holds integrated, global variables
    - this might be helpful to implement Lagrange multipliers and the like
    - generally, we should use a state class that contains the state of a pde
      (in most cases, this would be a FieldBase)
    - the state class should also handle serialization and io with hdf