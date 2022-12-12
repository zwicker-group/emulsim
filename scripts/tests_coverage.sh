#!/bin/bash

# add the likely paths of custom packages, relative to current base path
export PYTHONPATH=submodules/py-pde:submodules/py-droplets:submodules/py-phasesep:submodules/py-modelrunner:$PYTHONPATH

echo 'Determine coverage of all unittests...'

./run_tests.py --unit --coverage --nojit --parallel