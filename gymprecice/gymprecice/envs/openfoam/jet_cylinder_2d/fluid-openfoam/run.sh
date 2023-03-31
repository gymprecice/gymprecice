#!/bin/bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
#------------------------------------------------------------------------------
# run case
mpirun -np $(getNumberOfProcessors) --bind-to none  pimpleFoam  -parallel > log.pimpleFoam 2>&1 &
