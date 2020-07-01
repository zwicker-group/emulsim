TODO
====
* Use black style
* Update readme
* Improve plotting
    - adjust color of droplets
    - adjust bounds to correct background field
* Implement MeanfieldElement by specializing ScalarFieldElement
* Documentation
* Convenient I/O for states and simulations (actor + couplings)
* Add napari plotting
* Add graph representation to simulation connections
* Allow easy addition of the simulation parameters when writing the state
  to a file (for documentation purposes)
* AgentSimulation:
	Add diagnostic information (dt, step_count, agent_count)
* Add repr and str methods to agents and background classes
* Improve analysis and post-processing
    # analysis
    result.get_quantity('total_amount')
    result.plot()
    result.to_file('result.hdf5')
    
    loaded = State.from_file('result.hdf5')
