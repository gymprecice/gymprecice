#!/bin/bash
cd "${0%/*}" || exit
. ${DRL_BASE:?}/openfoam/RunFunctions
#------------------------------------------------------------------------------
echo ${DRL_BASE}
# run case
mpirun -np 2 --bind-to none  pimpleFoam  -parallel > log.pimpleFoam 2>&1 &
