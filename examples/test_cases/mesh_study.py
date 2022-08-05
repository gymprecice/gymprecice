from asyncio import subprocess
from os.path import join
import os
from shutil import copytree
from copy import deepcopy
import subprocess
import argparse
from os import makedirs


def parseArguments():

    ag = argparse.ArgumentParser()
    ag.add_argument('-d', '--base_directory', help='Specify base_case directory', required=False, default='.')
    ag.add_argument('-b','--base_name', help='Specify base_case directory', required=False, default='base')
    ag.add_argument('-m', '--mesh_resolution', help='Specify mesh resolution for the study', type=int, nargs='+', required=True)
    ag.add_argument('-n', '--n_procs', help='Specify number of procs for each study', type=int, nargs='+', required=True)
    ag.add_argument('-s', '--save_directory', help='Specify directory name to save runs', required=False, default='mesh_study')
    ag.add_argument('-t', '--end-time', type=float,  help='Specify end time of simulation', required=False, default=5.0)
    args = ag.parse_args()
    return args

def main(args):

    # setting
    base_path = args.base_directory
    base_name = args.base_name
    mesh_resolution = args.mesh_resolution
    n_procs = args.n_procs
    end_time = args.end_time
    save_directoty = args.save_directory

    save_path = join(base_path, save_directoty)
    makedirs(save_path, exist_ok=True)

    for idx, case in enumerate(mesh_resolution):
        cwd = join(save_path, f'mesh{case}')
        copytree(join(base_path, base_name), cwd, dirs_exist_ok=True)
        cmd_str = f'./mesh_study {case} {n_procs[idx]} {end_time} > log.mesh_run 2>&1'
        subprocess.Popen(cmd_str, shell=True, cwd=cwd)


if __name__ == "__main__":
    args = parseArguments()
    main(args)