# agent-based-emulsions

[![Build status](https://github.com/zwicker-group/agent-based-emulsions/workflows/build/badge.svg)](https://github.com/zwicker-group/agent-based-emulsions/actions?query=workflow%3Abuild)

Agent-based model for simulating emulsions efficiently.



Main idea
---------

The idea is to have a AgentSimulation class that can be controlled to run
simulations. This evolves and AgentState, which contains all the instantaneous
information about the simulation. AgentState consists of a single instance
handling the background and an AgentCollection that contains all (potentially
different) agents. AgentCollection is a list of AgentsBase, which collects
agents of identical behavior. 


Installation
------------

Since this package is not public, it cannot be installed using `pip`. Instead
the repository needs to be cloned from github. The necessary python packages
can be installed using `pip`. To install the package together with the
requirements, the following commands can be used:

```bash
git clone https://github.com/zwicker-group/agent-based-emulsions.git
pip install -r agent-based-emulsions/requirements.txt
```


Documentation
-------------

The documentation for this package is not publicly available, but it can be
generated from the source code. To do this, additional requirements have to be
installed and the build script has to be called from the `docs` directory.
The following commands, run from the root directory of the repository, achieve
this:

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

The main entry point to the documentation is then the webpage
`docs/build/html/index.html`.


Running tests
-------------

The package comes with automated tests that reside in `tests` directories in the
respective python packages. The purpose of these tests is to ensure some basic
functionality of the package. Consequently, it is good practise to run the tests
and fix problems before commiting to the repository. There are a number of
convenient scripts collected in the root `tests` directory that can be used for
this. In particular, there is a `requirements.txt` for installing the necessary
python components:
 
```bash
pip install -r tests/requirements.txt
```

The actual scripts in the `tests` directory servere different purposes:

* `tests_run.sh` runs all tests in sequential order. The script takes an
  optional argument that selects which tests are run: Only test files or methods
  that match the argument will be run.
* `tests_parallel.sh` runs all tests in parallel. Also takes a pattern argument.
* `codestyle.sh` tests whether the code style is obeyed by all files. Problems
  in the code style should be resolved to achieve a uniform experience for
  everyone.
* `tests_types.sh` tests the type annotations in the python files. Type
  annotations are optional in python, but they can be helpful to spot subtle
  programming problems. 
