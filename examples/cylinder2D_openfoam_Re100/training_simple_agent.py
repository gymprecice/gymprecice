from pickle import TRUE
import gym
from OpenFoamRLEnv import OpenFoamRLEnv
from utils import fix_randseeds
import numpy as np
import time
import torch.nn as nn

from agent import SimpleAgent


def make_env():
    # shell options to run the solver (this can/should be placed in a
    # separate python script)
    foam_case_path = "cylinder2D-structured-mesh"
    foam_shell_cmd = "foam-functions-cylinder2D.sh"
    foam_clean_cmd = "cleanfoam"
    foam_softclean_cmd = "softcleanfoam"
    foam_prerunclean_cmd = "preruncleanfoam"

    foam_preprocess_cmd = "preprocessfoam" 
    foam_run_cmd = "runfoam"
    foam_prerun_cmd = "prerunfoam"
    foam_preprocess_log = "foam_preprocess.log"
    foam_clean_log = "foam_clean.log"
    foam_softclean_log = "foam_softclean.log"
    foam_prerunclean_log = "foam_prerun_clean.log"
    foam_run_log = "foam_run.log"
    foam_prerun_log = "foam_prerun.log"

    parallel_run = False
    if parallel_run:
        foam_preprocess_cmd += " -parallel"
        foam_run_cmd += " -parallel"

    # if True, then the preprocessing (here: blockMesh) happens per each epoch:
    foam_full_reset = False

    foam_clean_cmd = f" && {foam_clean_cmd} > {foam_clean_log} 2>&1"
    foam_softclean_cmd = f" && {foam_softclean_cmd} > {foam_softclean_log} 2>&1"
    foam_prerunclean_cmd = f" && {foam_prerunclean_cmd} > {foam_prerunclean_log} 2>&1"
    foam_preprocess_cmd = f" && {foam_preprocess_cmd} > {foam_preprocess_log} 2>&1"
    foam_run_cmd = f" && {foam_run_cmd} > {foam_run_log} 2>&1"
    foam_prerun_cmd = f" && {foam_prerun_cmd} > {foam_prerun_log} 2>&1"

    # reset options
    n_trajectories = 10
    # Size and type is redundant data (included controlDict or called from a file)
    # Multiple way to do this in OpenFoam so we delegate it to user
    postprocessing_data = {
        'forces': {
            'use': 'reward',  # goes into observation or rewards
            'type': 'forces',  # forces|probe|?
            'datatype': 'scaler',  # scaler vs field
            'size': 12,  # number of forces
            'prerun_output_file': '/postProcessing/forces/0/coefficient.dat',  # depends on the type of the probe/patchProbe/etc
            'output_file': '/postProcessing/forces/0.01/coefficient.dat',  # depends on the type of the probe/patchProbe/etc
        },
        'p': {
            'use': 'observation',  # goes into observation or rewards
            'type': 'probe',  # forces|probe|?
            'datatype': 'scaler',  # scaler vs field
            'size': 11,  # number of probes
            'prerun_output_file': '/postProcessing/probes/0/p',  # depends on the type of the probe/patchProbe/etc
            'output_file': '/postProcessing/probes/0.01/p',  # depends on the type of the probe/patchProbe/etc
        },
        # 'U': {
        #     'use': 'observation',  # goes into observation or rewards
        #     'type': 'probe',  # forces|probe|?
        #     'datatype': 'field',  # scaler vs field
        #     'size': 3,  # number of probes
        #     'output_file': '/postProcessing/patchProbes/0/U',
        # },
        # 'T': {
        #     'use': 'observation',  # goes into observation or rewards
        #     'type': 'probe',  # forces|probe|?
        #     'datatype': 'scaler',  # scaler vs field
        #     'size': 3,  # number of probes
        #     'output_file': '/postProcessing/patchProbes/0/T',
        # },
    }

    options = {
        "precice_cfg": "precice-config.xml",
        "case_path": foam_case_path,
        "foam_shell_cmd": foam_shell_cmd,
        "clean_cmd": foam_clean_cmd,
        "softclean_cmd": foam_softclean_cmd,
        "prerunclean_cmd": foam_prerunclean_cmd,
        "preprocess_cmd": foam_preprocess_cmd,
        "prerun_cmd": foam_prerun_cmd,
        "run_cmd": foam_run_cmd,
        "solver_full_reset": foam_full_reset,
        "rand_seed": rand_seed,
        "postprocessing_data": postprocessing_data,
        "n_parallel_env": n_trajectories,
        "prerun_needed": True,
        "is_dummy_run": False
    }

    # create the environment
    # env = gym.make("FoamAdapterEnv-v0")
    env = OpenFoamRLEnv(options)

    return env

if __name__ == '__main__':
    rand_seed = 12345
    fix_randseeds(rand_seed)

    agent = SimpleAgent(lambda : make_env())

    mlp_params = {
        'hidden_sizes': [64, 64],
        'activation': nn.Tanh,
        'output_activation': nn.Tanh
    }

    train_params = {
        'epochs': 3,
        'batch_size': 20
    }

    print('\nUsing simplest formulation of policy gradient.\n')
    agent.train(mlp_params, train_params)
