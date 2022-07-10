#!/bin/sh

DIR=$(cd "$(dirname "${BASH_SOURCE}")" > /dev/null 2>&1 && pwd )

preprocess() {
    set -e -u
    cd "$DIR${1#.}"
    echo "-- Running OF-blockMesh in $(pwd) ..." 
    blockMesh
    touch fluid-openfoam.foam
}

run() {
    set -e +u
    cd "$DIR${1#.}"
    echo "-- Running OF-solver in $(pwd) ..."
    . "${WM_PROJECT_DIR}/bin/tools/RunFunctions"
    solver=$(getApplication)
    if [ "${2:-}" = "-parallel" ]; then
        procs=$(getNumberOfProcessors)
        decomposePar -force
        mpirun -np "${procs}" "${solver}" -parallel
        reconstructPar
    else
        ${solver}
    fi 
}

clean() {
    set -e -u
    cd "$DIR${1#.}"
    echo "--- Cleaning up OF-case in $(pwd)"
    if [ -n "${WM_PROJECT:-}" ] || error "No OpenFOAM environment is active."; then
        # shellcheck disable=SC1090 # This is an OpenFOAM file which we don't need to check
        . "${WM_PROJECT_DIR}/bin/tools/CleanFunctions"
        cleanCase
        rm -rfv 0/uniform/functionObjects/functionObjectProperties
    fi
    rm -rfv ./preCICE-output/ \
            ./precice-*-iterations.log \
            ./precice-*-convergence.log \
            ./precice-*-events.json \
            ./precice-*-events-summary.log \
            ./precice-postProcessingInfo.log \
            ./precice-*-watchpoint-*.log \
            ./precice-*-watchintegral-*.log \
            ./core \
            log.*
}