#!/bin/sh

preprocessfoam() {
    set -e 
    . "${WM_PROJECT_DIR}/bin/tools/RunFunctions"
    
    # dummy files for post-processing with paraview
    touch case.foam
    touch case.OpenFOAM

    # mesh creation
    runApplication blockMesh
    runApplication transformPoints -translate '(-0.2 -0.2 0)' 
    runApplication topoSet
    runApplication createPatch -overwrite
    runApplication renumberMesh -overwrite

    # set inlet velocity
    # cp -r 0.org 0
    cp -r start_0 0

    runApplication setExprBoundaryFields

    if [ "${1-}" = "-parallel" ]; then
        # decompose case
        runApplication decomposePar
        runParallel renumberMesh -overwrite
    fi
}

runfoam() {
    set -e
    . "${WM_PROJECT_DIR}/bin/tools/RunFunctions"
    # soft cleaning removed the sybmolic links to case folder
    # create real files for reach run
    touch case.foam
    touch case.OpenFOAM

    if [ "${1-}" = "-parallel" ]; then
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
            processor* \
            log.* \
            *.json \
            *.log \
            *.foam \
	        system/*.msh \
            *.OpenFOAM
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
    rm -rfv ./processor*/0.* 
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
            *.log \
            *.foam \
	        system/*.msh \
            *.OpenFOAM
}

preprocessUnstructuredFoam() {
    set -e 
    . "${WM_PROJECT_DIR}/bin/tools/RunFunctions"
    
    # dummy files for post-processing with paraview
    touch case.foam
    touch case.OpenFOAM

    # mesh creation
    gmsh -3 "system/unstructured_msh.geo" > log.gmsh 2>&1
    runApplication gmshToFoam -noFunctionObjects "system/unstructured_msh.msh"
    sed -i '/physicalType/d' ./constant/polyMesh/boundary
    awk -v RS="" '{ gsub(/front\n    {\n        type            patch/, "front\n    {\n        type            empty"); print }' constant/polyMesh/boundary > tmp 
    mv tmp constant/polyMesh/boundary
    awk -v RS="" '{ gsub(/back\n    {\n        type            patch/, "back\n    {\n        type            empty"); print }' constant/polyMesh/boundary >tmp
    mv tmp constant/polyMesh/boundary
    
    runApplication topoSet
    runApplication createPatch -overwrite
    runApplication renumberMesh -overwrite

    # set inlet velocity
    cp -r 0.org 0
    runApplication setExprBoundaryFields

    if [ "${1-}" = "-parallel" ]; then
        # decompose case
        runApplication decomposePar
        runParallel renumberMesh -overwrite
    fi
}
