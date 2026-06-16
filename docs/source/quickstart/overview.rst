Overview
^^^^^^^^

The main aim of the :mod:`emulsim` package is to simulate the dynamics of coupled
systems.
In these simulations, several `elements` define the dynamic parts that will
evolve over time.
The actual dynamics are defined by `actors`, which can act on either a single
element or couple multiple elements.

Alternative python packages that pursue similar goals include:

* :mod:`simframe`: `Source code <https://github.com/stammler/simframe/>`_ and `documentation <https://simframe.readthedocs.io/>`_
* :mod:`PySim`: `Source code <https://github.com/aldebjer/pysim>`_ and `documentation <https://pysim.org>`_. Focuses on ordinary differential equations.