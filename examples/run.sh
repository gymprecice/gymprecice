#!/bin/bash 

set -e -u

python3  ./test.py ./precice-config.xml > log.adapter  2>&1 &
python3  ./pythonFluidSolver/FluidSolver.py ./precice-config.xml > log.solver 2>&1 &
