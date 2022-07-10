import subprocess

subprocess.Popen("blockMesh && touch fluid-openfoam.foam && ../tools/run-openfoam.sh \"$@\" && . ../tools/openfoam-remove-empty-dirs.sh && openfoam_remove_empty_dirs", shell=True)
