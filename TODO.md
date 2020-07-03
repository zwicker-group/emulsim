TODO
====
* Improve plotting
    - adjust color of droplets
* Convenient I/O for states and simulations (actor + couplings)
* Add napari plotting
* Allow easy addition of the simulation parameters when writing the state
  to a file (for documentation purposes)
* Simulation:
	Add diagnostic information (dt, step_count, degrees of freedom)
* Improve analysis and post-processing
    # analysis
    result.get_quantity('total_amount')
    result.plot()
    result.to_file('result.hdf5')
    
    loaded = State.from_file('result.hdf5')
