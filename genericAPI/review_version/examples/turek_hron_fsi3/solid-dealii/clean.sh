#!/bin/sh
set -e -u

echo "--- Cleaning up deal.II case in $(pwd)"
rm -rfv ./dealii-output/

echo "---- Cleaning up preCICE logs in $(pwd)"
rm -fv ./precice-*-iterations.log \
./precice-*-convergence.log \
./precice-*-events.json \
./precice-*-events-summary.log \
./precice-postProcessingInfo.log \
./precice-*-watchpoint-*.log \
./precice-*-watchintegral-*.log \
./core
