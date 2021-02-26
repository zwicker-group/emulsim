Running unit tests
^^^^^^^^^^^^^^^^^^

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

