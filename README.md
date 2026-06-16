# emulsim

Framework for simulating physical systems consisting of multiple, interacting entities.
The current focus of the package lies on simulating interacting droplets in a common
environment, but the framework is extensible and flexible.


## Main idea

The main idea is that the physical system consists of multiple `elements`, which
together describe the state of the system. The dynamical rules are encoded in `actors`,
which either act on individual elements, encoding their autonomous dynamics, or on
multiple elements, introducing couplings.


## Installation

The package is not yet available on `pip` and must thus be cloned from Github. The
necessary python packages can be installed using `pip`. To install the package together
with the requirements, the following commands can be used:

```bash
git clone https://github.com/david-zwicker/emulsim.git
pip install -r emulsim/requirements.txt
```

## Documentation

The [documentation for this package is available online](https://emulsim.readthedocs.io/)


## Running tests

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
