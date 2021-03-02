Coding style
^^^^^^^^^^^^
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