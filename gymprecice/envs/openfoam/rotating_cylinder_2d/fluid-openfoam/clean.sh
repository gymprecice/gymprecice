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
    # cleanPostProcessing
    # cleanTimeDirectories
    rm -rf ./preCICE-output/
    rm -rf ./preCICE-*/
    rm -rf  ./postProcessing/*/*[1-9]*
    rm -f log.*
    #decomposePar -force
)