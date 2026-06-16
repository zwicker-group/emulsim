Data model
^^^^^^^^^^

The main idea of the :mod:`emulsim` package is to separate the description of the *state*
of the simulation from the description that govern the dynamics.
The state gives all the necessary information to represent the system at a particular
time point.
Since we restrict ourself to memory-less systems (i.e., Markov chains), the state
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
:class:`~emulsim.state.State`. All elements defined by the package are collected in the
:ref:`list-of-all-elements`.


Each element has internal degrees of freedom, which can change over time and can
accessed by the :attr:`~emulsim.elements.base.ElementBase.data` attribute.
Elements might also have :attr:`~emulsim.elements.base.ElementBase.attributes` that do not
change over time. 
Attributes are typically given when initializing an element.
For instance, most elements accept an attribute `parameters`, which defines the physical
parameters like the mass of an object.
Together, :attr:`~emulsim.elements.base.ElementBase.data` and
:attr:`~emulsim.elements.base.ElementBase.attributes` fully describe an element.
In particular, those two elements should contain all information to re-create the
element, e.g., from a file.
To summarize, the key assumption of :mod:`emulsim` is that elements have fixed properties,
described by attributes, and dynamical degrees of freedom, described by data.

Some elements represent collections of identical items, e.g., the
:class:`~emulsim.elements.spherical_droplets.SphericalDropletsElement`. In this case, the
:attr:`data` attribute is an array specifying the degree of freedoms for each item.
Since each individual item could have multiple degrees of freedom (e.g., position and
radius for the a droplet), we use structured datatypes of numpy, which can be created
with the :class:`numpy.dtype` class. 

To define a custom element, you need to define a class that inherits from
:class:`~emulsim.elements.base.ArrayElementBase`,
:class:`~emulsim.elements.base.ArrayCollectionElementBase`, or
:class:`~emulsim.elements.base.ObjectElementBase` depending on what data is best describing
the degrees of freedom of your element.
These base elements can already support the data attribute and are fully functional.
To customize the element, you can add model parameters to it by defining the class
attribute `parameters_default`.
If you want to add attributes other than parameters, you need to overwrite the
:attr:`~emulsim.elements.base._ElementBase.attributes` property and the class method
:meth:`~emulsim.elements.base._ElementBase._init_state`, which initializes objects from a 
supplied state.
These aspects are explained in the code example below:

.. code-block:: python

    from emulsim.elements.base import ElementBase

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

        def _init_state(self, attributes, data):
            super()._init_state(attributes, data)
            self.name = attributes.get("name", "No name")


Simulation dynamics (Actors)
############################
The system state changes according to physical principles.
Since the state is encoded in *elements*, the physics must change the dynamical degrees
of freedom of the elements, i.e., their :attr:`data` attributes.


In the :mod:`emulsim` package, the dynamics are defined by a 
:class:`~emulsim.simulation.Simulation` object.
Each simulation contains one or more *actors*, which each affect one or multiple
elements.
All actors defined by the package are collected in the :ref:`list-of-all-actors`.
This approach allows combining multiple actors without redefining their code simply by
combining them in a :class:`~emulsim.simulation.Simulation`. 
Each actor inherits from :class:`~emulsim.actors.base.ActorBase`, which defines the necessary
behavior.
In the simplest case, a custom actor only needs to overwrite the
:meth:`~emulsim.actors.base.ActorBase.evolve` method, which evolves its elements from time
:code:`t` to :code:`t + dt`, changing the respective :attr:`data` attributes in place:

.. code-block:: python

    class BrownianParticlesActor(emulsim.ActorBase):

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
