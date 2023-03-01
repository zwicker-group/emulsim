#!/usr/bin/env bash

# set the likely paths for the pde and droplet package for local testing
export MYPYPATH=submodules/py-pde:submodules/py-phasesep:submodules/py-droplets:$MYPYPATH

./run_tests.py --types 
