Data model
^^^^^^^^^^

The main idea of the :mod:`sim` package is to separate the description of the *state*
of the simulation from the description that govern the dynamics.
The state gives all the necessary information to represent the system at a particular
time point.
Since we restrict ourself to memoryless systems (i.e., Markov chains), the state
contains all information to evolve the system forward in time.
The full dynamics then leads to a sequence of states, collectively called a
*trajectory*.
The aim of the package is to evolve an initial state according to some dynamics to
obtain the entire trajectory or only the final state.


System state (Elements)
#######################
A full physical system consist of *elements* that define *what* is present in the system
and *actors* that define *how* these elements interact and evolve in time.
All the elements in a system together described by the system's
:class:`~sim.state.State`.

Each element has internal degrees of freedom, which can change over time and can
accessed by the :attr:`~sim.elements.base.ElementBase.data` attribute.
Elements might also have :attr:`~sim.elements.base.ElementBase.attributes` that do not
change over time. 
Attributes are typically given when initializing an element.
For instance, most elements accept an attribute `parameters`, which defines the physical
parameters like the mass of an object.
Together, :attr:`~sim.elements.base.ElementBase.data` and
:attr:`~sim.elements.base.ElementBase.attributes` fully describe an element.
In particular, those two elements should contain all information to re-create the
element, e.g., from a file.
To summarize, the key assumption of :mod:`sim` is that elements have fixed properties,
described by attributes, and dynamical degrees of freedom, described by data.

To define a custom element, you need to define a class that inherits from
:class:`~sim.elements.base.ElementBase`. This default element can already take a 
:class:`~numpy.ndarray` as the data attribute and is fully functional.
To customize the element, you can add model parameters to it by defining the class
attribute `parameters_default`.
If you want to add attributes other than parameters, you need to overwrite the
:attr:`~sim.elements.base.ElementBase.attributes` property and the class method
:meth:`~sim.elements.base.ElementBase.from_state`, which initializes objects from a 
supplied state.
These aspects are explained in the code example below:

.. code-block:: python
    
    from sim.elements.base import ElementBase
    
    class CustomElement(ElementBase):
        
        parameters_default = {'mass': 10}
        
        def __init__(self, data, name="Custom", parameters=None):
            super().__init__(data, parameters)
            self.name = name
        
        @property
        def attributes(self):
            attrs = super().attributes
            attrs['name'] = self.name
            return attrs
            
        @classmethod
        def from_state(cls, attributes, data=None):
            obj = super().from_state(attributes, data)
            obj.name = attributes.get("name", "No name")
            return obj        


If an attribute value is a custom object, you might also need to overwrite the 
:meth:`~sim.elements.base.ElementBase.serialize_attribute` and
:meth:`~sim.elements.base.ElementBase.unserialize_attribute` methods to define how the object can
be converted to a string representation and vice versa.
Moreover, it will usually be helpful to overwrite the
:meth:`~sim.elements.base.ElementBase.plot` method to allow displaying the element.
These three methods are quickly showcased in the following snippet:

.. code-block:: python
    
    class CustomElement(ElementBase):
        
        [...]
        
        def serialize_attribute(self, name, value):
            if name == 'complicated_attribute':
                # treat the special attribute
                return value.get_string_representation()
                
            # fall back to default behavior for all others
            return super().serialize_attribute(name, value)

        @classmethod
        def unserialize_attribute(cls, name, value_str):
            if name == 'complicated_attribute':
                # treat the special attribute
                return complicated_attribute_from_str(value_str)

            # fall back to default behavior for all others
            return super().unserialize_attribute(name, value_str)
    
        def plot(self, ax=None, *args, **kwargs):
            ax.plot(self.data)



Simulation dynamics (Actors)
############################
The system state changes according to physical principles.
Since the state is encoded in *elements*, the physics must change the dynamical degrees
of freedom of the elements, i.e., their :attr:`data` attributes.


In the :mod:`sim` package, the dynamics are defined by a 
:class:`sim.simulation.Simulation` object.
Each simulation contains one or more *actors*, which each affect one or multiple
elements.
This approach allows combining multiple actors without redefining their code simply by
combining them in a :class:`sim.simulation.Simulation`. 
Each actor inherits from :class:`sim.actors.base.ActorBase`, which defines the necessary
behavior.
In the simplest case, a custom actor only needs to overwrite the
:meth:`sim.actors.base.ActorBase.evolve` method, which evolves its elements from time
:code:`t` to :code:`t + dt`, changing the respective :attr:`data` attributes in place:

.. code-block:: python
    
    class BrownianParticlesActor(sim.ActorBase):
    
        diffusivity = 1
    
        def evolve(self, elements, t, dt):
            """ evolve the particles in time """
            (particles,) = elements
            scale = np.sqrt(dt) * self.diffusivity
            particles.data[...] += scale * np.random.normal(size=particles.data.shape)
    
        
        def make_evolver_numba(self, elements):
            """return a function evolve the field state from time `t` to `t + dt` """
            diffusivity = self.diffusivity
    
            @jit
            def evolver(state_data, t, dt):
                """ evolve all points explicitly """
                scale = np.sqrt(dt * diffusivity)
                for i in range(state_data[0].size):
                    state_data[0].flat[i] += scale * np.random.randn()
    
            return evolver
