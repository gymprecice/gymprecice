#!/bin/sh
set -e -u

solver_init()
blockMesh
touch fluid-openfoam.foam

../tools/run-openfoam.sh "$@"
. ../tools/cleaning-tools.sh && clean_openfoam .

