#!/bin/sh

preprocessfoam() {
    set -e 
    . "${WM_PROJECT_DIR}/bin/tools/RunFunctions"
    
    # paraview reader
    touch fluid-openfoam.OpenFOAM
    # parafoam reader
    touch fluid-openfoam.foam
    
    # mesh creation
    runApplication blockMesh
    runApplication transformPoints -translate '(-0.2 -0.2 0)' 

    # set inlet velocity
    cp -r 0.org 0
    runApplication setExprBoundaryFields

    if [ "${1:-}" = "-parallel" ]; then
        # decompose and run case
        runApplication decomposePar
        runParallel renumberMesh -overwrite
    fi
}

runfoam() {
    set -e
    . "${WM_PROJECT_DIR}/bin/tools/RunFunctions"

    if [ "${1:-}" = "-parallel" ]; then
        echo "-- OpenFoam solver $(getApplication) ... parallel run"
        runParallel $(getApplication)
    else
        echo "-- OpenFoam solver $(getApplication) ... serial run"
        runApplication $(getApplication)
    fi 
}

cleanfoam() {
    set -e
    . "${WM_PROJECT_DIR}/bin/tools/CleanFunctions"
    cleanCase
    rm -rfv 0
    rm -rfv ./preCICE-output/ \
            ./precice-*-iterations.log \
            ./precice-*-convergence.log \
            ./precice-*-events.json \
            ./precice-*-events-summary.log \
            ./precice-postProcessingInfo.log \
            ./precice-*-watchpoint-*.log \
            ./precice-*-watchintegral-*.log \
            ./core \
            ./postProcessing \
            log.* \
            *.json \
            *.log
}

softcleanfoam() {
    set -e
    . "${WM_PROJECT_DIR}/bin/tools/CleanFunctions" 
    cleanTimeDirectories
    cleanAdiosOutput
    cleanAuxiliary
    cleanDynamicCode
    cleanOptimisation
    cleanPostProcessing

    rm -rfv ./preCICE-output/ \
            ./precice-*-iterations.log \
            ./precice-*-convergence.log \
            ./precice-*-events.json \
            ./precice-*-events-summary.log \
            ./precice-postProcessingInfo.log \
            ./precice-*-watchpoint-*.log \
            ./precice-*-watchintegral-*.log \
            ./core \
            ./postProcessing \
            log.* \
            *.json \
            *.log
}