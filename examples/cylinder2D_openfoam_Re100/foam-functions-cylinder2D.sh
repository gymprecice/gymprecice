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

    # check argumets
    parallel_run=false
    pre_run=false

    for arg in "$@"; do
        case $arg in
            "-parallel") parallel_run=true ;;
            "-prerun") pre_run=true ;;
            *) ;;
        esac
    done

    # turn off preCICE adapter for "prerun" 
    if [ "$pre_run" = "true" ]; then
        sed  -i "s/^\s\{0,\}\(\#.*preCICE_Adapter.*\)/\/\/ \1 /g" ./system/controlDict
        for arg in "$@"; do
            if [ "$arg" = "-prerun" ] || [ "$arg" = "-parallel" ]; then
                shift
            fi
        done
        sed -i "s/^\s\{0,\}endTime.*/endTime    ${1};/g" ./system/controlDict
        sed -i "s/^\s\{0,\}writeControl.*/writeControl   timeStep;/g" ./system/controlDict
    else
        sed -i "s/^\s\{0,\}\/\/.*preCICE_Adapter.*/    #includeFunc preCICE_Adapter/g" ./system/controlDict
        sed -i "s/^\s\{0,\}endTime.*/endTime    1000;/g" ./system/controlDict
        sed -i "s/^\s\{0,\}writeControl.*/writeControl   none;/g" ./system/controlDict
    fi

    if [ "$parallel_run" = "true" ]; then
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
    # cleanCase
    cleanTimeDirectories
    cleanAuxiliary
    cleanDynamicCode
    cleanOptimisation
    
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
    cleanAdiosOutput
    cleanAuxiliary
    cleanDynamicCode
    cleanOptimisation
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
            log.* \
            *.json \
            *.log \
            *.foam \
	        system/*.msh \
            *.OpenFOAM
    
    pre_run=false
    for arg in "$@"; do
        if [ "$arg" = "-prerun" ]; then
            pre_run=true
            break
        fi
    done

    if [ "$pre_run" = "true" ]; then
        rm -rfv  ./postProcessing/*/*[1-9]*
    else
        cleanTimeDirectories
        cleanPostProcessing
        rm -rfv ./processor*/0.* \
                ./postProcessing 
    fi 
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
