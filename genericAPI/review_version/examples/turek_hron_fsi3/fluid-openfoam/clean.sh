#!/bin/bash
cd "${0%/*}" || exit
set -e
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions 

#------------------------------------------------------------------------------

(
    cleanAdiosOutput
    cleanAuxiliary
    cleanDynamicCode
    cleanOptimisation
    cleanPostProcessing
    cleanTimeDirectories
    rm -rf ./processor*
    rm -rf ./preCICE-output/
    rm -rf ./preCICE-*/
    rm -f ./precice-*-iterations.log \
    ./precice-*-convergence.log \
    ./precice-*-events.json \
    ./precice-*-events-summary.log \
    ./precice-postProcessingInfo.log \
    ./precice-*-watchpoint-*.log \
    ./precice-*-watchintegral-*.log \
    ./core
)
