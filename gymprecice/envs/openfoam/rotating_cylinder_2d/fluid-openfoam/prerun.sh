#!/bin/bash
cd "${0%/*}" || exit
set -e
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions 

#------------------------------------------------------------------------------

(
    cleanAdiosOutput
    cleanAuxiliary
    cleanDynamicCode
    cleanOptimisation
    #cleanPostProcessing
    #cleanTimeDirectories
    rm -f log.*
    rm -rf ./preCICE-output/
    rm -rf ./preCICE-*/
 
    runApplication setExprBoundaryFields
    runApplication decomposePar -force
    runParallel renumberMesh -overwrite
)
