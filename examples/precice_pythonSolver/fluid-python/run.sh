#!/bin/sh
set -e -u

python3 ./FluidSolver.py ../precice-config.xml > log.solver 2>&1 &
