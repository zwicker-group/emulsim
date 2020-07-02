Contributing code
^^^^^^^^^^^^^^^^^


Structure of the package
""""""""""""""""""""""""
The functionality of the :mod:`sim` package is split into multiple
sub-package. In particular, there are classes representing different elements
and different actors, which can all be extended individually.


Coding style
""""""""""""
In terms of coding style, we try to adhere to `PEP8
<https://www.python.org/dev/peps/pep-0008/>`_ and use `Google Style docstrings
<https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings>`_.
The docstring convention might be best `learned by example
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
The documentation, including the docstrings, are written using `reStructuredText
<https://de.wikipedia.org/wiki/ReStructuredText>`_, with examples in the
following `cheatsheet
<https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_.
To ensure the integrity of the code, we also try to provide many test functions,
which are typically contained in separate modules in sub-packages called
:mod:`tests`.
These tests can be ran using scripts in the :file:`tests` subfolder in the root
folder.
This folder also contain a script :file:`tests_types.sh`, which uses :mod:`mypy`
to check the consistency of the python type annotations.
We use these type annotations for additional documentation and they have also
already been useful for finding some bugs.

We also have some conventions that should make the package more consistent and
thus easier to use. For instance, we try to use ``properties`` instead of getter
and setter methods as often as possible.
Because we use a lot of :mod:`numba` just-in-time compilation to speed up computations,
we need to pass around (compiled) functions regularly. The names of the methods
and functions that make such functions, i.e. that return Callables, should start
with 'make_*' where the wildcard should describe the purpose of the function
being created. 


Running unit tests
""""""""""""""""""
The :mod:`sim` package contains several unit tests, typically contained
in sub-module :mod:`tests` in the folder of a given module. These tests ensure
that basic functions work as expected, in particular when code is changed in
future versions. To run all tests, there are a few convenience scripts in the
root directory :file:`tests`. The most basic script is :file:`tests_run.sh`,
which uses :mod:`pytest` to run the tests in the sub-modules of the
:mod:`sim` package. Clearly, the python package :mod:`pytest` needs to
be installed. There are also additional scripts that for instance run tests in
parallel (need the python package :mod:`pytest-xdist` installed), measure test 
coverage (need package :mod:`pytest-cov` installed), and make simple performance
measurements. Moreover, there is a script :file:`test_types.sh`, which uses
:mod:`mypy` to check the consistency of the python type annotations and there is
a script :file:`codestyle.sh`, which checks the coding style.

Before committing a change to the code repository, it is good practice to run
the tests, check the type annotations, and the coding style with the scripts
described above.

