Data model
^^^^^^^^^^

The main idea of the :mod:`sim` package is to separate the description of the *state*
of the simulation from the description that govern the dynamics.
The state gives all the necessary information to represent the system at a particular
time point. The full dynamics then leads to a sequence of states, collectively called a
*trajectory*.
The aim of the package is to evolve an initial state according to some dynamics to
obtain the entire trajectory or only the final state.
Below, we discuss how the state is structure and we briefly touch on how the dynamics
are described.


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
:meth:`~sim.elements.base.serialize_attribute` and
:meth:`~sim.elements.base.unserialize_attribute` methods to define how the object can
be converted to a string representation and vice versa.



Simulation dynamics (Actors)
############################