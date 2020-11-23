#!/bin/bash

# add the likely paths of custom packages, relative to current base path
export PYTHONPATH=../py-pde:../py-droplets:../py-phasesep:$PYTHONPATH

echo 'Determine coverage of all unittests...'

./run_tests.py --unit --coverage --no_numba --parallel