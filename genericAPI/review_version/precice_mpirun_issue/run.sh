#!/bin/bash
cd "${0%/*}" || exit
set -e

mpirun -np 4 python3 dummy_solver.py