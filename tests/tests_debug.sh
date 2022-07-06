#!/bin/bash

# add the likely paths of custom packages, relative to current base path
export PYTHONPATH=submodules/py-pde:submodules/py-droplets:submodules/py-phasesep:$PYTHONPATH

if [ ! -z $1 ] 
then 
    # test pattern was specified 
    echo 'Run unittests with pattern '$1':'
    ./run_tests.py --unit --nojit --pattern "$1"
else
    # test pattern was not specified
    echo 'Run all unittests:'
    ./run_tests.py --unit --nojit
fi
