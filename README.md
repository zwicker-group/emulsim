# py-sim

[![Build status](https://github.com/zwicker-group/py-sim/workflows/build/badge.svg)](https://github.com/zwicker-group/py-sim/actions?query=workflow%3Abuild)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Framework for simulating physical systems consisting of multiple, interacting entities. 


Main idea
---------

The main idea is that the physical system consists of multiple `elements`, which
together describe the state of the system. The dynamical rules are encoded in `actors`,
which either act on individual elements, encoding their autonomous dynamics, or on
multiple elements, introducing couplings.


Installation
------------

Since this package is not public, it cannot be installed using `pip`. Instead
the repository needs to be cloned from github. The necessary python packages
can be installed using `pip`. To install the package together with the
requirements, the following commands can be used:

```bash
git clone https://github.com/david-zwicker/py-sim.git
pip install -r sim/requirements.txt
```

Note that the public packages `py-pde` and `py-droplets` are included as submodules in
the `submodules` folder. To update these, please run

```bash
git submodule update --init --recursive
git pull --recurse-submodules
```

after cloning the `py-sim`. To update submodules automatically every time, the following
git option is useful:

```bash
git config --global submodule.recurse true
```


Documentation
-------------

The documentation for this package is not publicly available, but it can be generated
from the source code. To do this, additional requirements have to be installed and the
build script has to be called from the `docs` directory. The following commands, run
from the root directory of the repository, achieve this:

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

The main entry point to the documentation is then the web page
`docs/build/html/index.html`.


Running tests
-------------

The package comes with automated tests that reside in `tests` directory. The purpose of
these tests is to ensure some basic functionality of the package. Consequently, it is
good practice to run the tests and fix problems before committing to the repository. To
run tests, first install the the necessary python modules:

```bash
pip install -r tests/requirements.txt
```

The tests can be run using pytest or using the convenient scripts collected in the
`scripts` directory:

* `tests_run.sh` runs all tests in sequential order. The script takes an
  optional argument that selects which tests are run: Only test files or methods
  that match the argument will be run.
* `tests_parallel.sh` runs all tests in parallel. Also takes a pattern argument.
* `tests_coverage.sh` checks how much of the code is covered by tests.
* `tests_types.sh` tests the type annotations in the python files. Type annotations are
  optional in python, but they can be helpful to spot subtle programming problems. 
* `format_code.sh` enforces the code style on all files.
