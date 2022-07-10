#!/bin/bash

set -e

rm -f ./fluid-python/log.solver
rm -f ./log.dummyTraining

python3 -u ./dummyTraining.py ./precice-config.xml > log.dummyTraining  2>&1
