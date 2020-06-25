TODO
====
* Update readme
* Add __all__ to modules imported by *
* Improve plotting
* Implement Parameter list serialization
* Improve state bases such that IO becomes easy
    - each class only needs to implement `to_serialized` and `from_serialized`
    - serialization returns Dict[str, str] for attributes and np.ndarray or
      Dict[str, np.ndarray] for data
    - ElementBase implements hdf5 IO
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


Extend the simulation framework beyond agent-based models
=========================================================
* Ingredients:
    - *Element: Classes describing different parts of a simulation
    - *Actors: Intrinsic dynamics of each part
    - *Coupling: Interaction between different parts
* State contains information about the state of all Parts
* State and actors and couplings should be organized in packages per topic:
    - background, droplets, etc.
    - the actors and the couplings need to check whether they supplied states are compatible
* Potential package name:
    - py-sim
* Pseudo code for using the package:
    state = State({'background': ScalarField(...),
                   'droplets': Emulsion(...)})
    sim = Simulation(state)
    sim.add_actor('background', PDEActor(...))
    sim.add_actor('droplets', ActiveDropletActor(...))
    sim.add_coupling('droplets', 'background', DropletBackgroundCoupling(...))
    result = sim.run(t_range=...)
    
    # analysis
    result.get_quantity('total_amount')
    result.plot()
    result.to_file('result.hdf5')
    
    loaded = State.from_file('result.hdf5')