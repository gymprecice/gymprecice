#!/bin/bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
#------------------------------------------------------------------------------
# run case

## parallel run
mpirun -np $(getNumberOfProcessors) --bind-to none  pimpleFoam  -parallel > log.pimpleFoam 2>&1 &

## single run
#runApplication $(getApplication)