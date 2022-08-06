import gym
from OpenFoamRLEnv import OpenFoamRLEnv
from utils import fix_randseeds
import numpy as np
import time


def agent(*argv):
    """
    dummy agent to return an action based on observation from env
    """
    env = argv[0]
    return env.action_space.sample()


if __name__ == '__main__':
    rand_seed = 12345
    fix_randseeds(rand_seed)

    # shell options to run the solver (this can/should be placed in a
    # separate python script)
    foam_case_path = "fluid-openfoam-cylinder2D"
    foam_shell_cmd = "foam-functions-cylinder2D.sh"
    foam_clean_cmd = "cleanfoam"
    foam_softclean_cmd = "softcleanfoam"

    foam_preprocess_cmd = "preprocessfoam" 
    foam_run_cmd = "runfoam"
    foam_preprocess_log = "foam_preprocess.log"
    foam_clean_log = "foam_clean.log"
    foam_softclean_log = "foam_softclean.log"
    foam_run_log = "foam_run.log"

    parallel_run = False
    if parallel_run:
        foam_preprocess_cmd += " -parallel"
        foam_run_cmd += " -parallel"

    # if True, then the preprocessing (here: blockMesh) happens per each epoch:
    foam_full_reset = False

    foam_clean_cmd = f" && {foam_clean_cmd} > {foam_clean_log} 2>&1"
    foam_softclean_cmd = f" && {foam_softclean_cmd} > {foam_softclean_log} 2>&1"
    foam_preprocess_cmd = f" && {foam_preprocess_cmd} > {foam_preprocess_log} 2>&1"
    foam_run_cmd = f" && {foam_run_cmd} > {foam_run_log} 2>&1"

    # foam_run_cmd = f"sh ../foam-run.sh > {foam_run_log} 2>&1"
    # foam_preprocess_cmd = f"sh ../foam-preprocess.sh > {foam_preprocess_log} 2>&1"

    # reset options
    n_parallel_env = 1

    # Size and type is redundant data (included controlDict or called from a file)
    # Multiple way to do this in OpenFoam so we delegate it to user
    postprocessing_data = {
        'forces': {
            'use': 'reward',  # goes into observation or rewards
            'type': 'forces',  # forces|probe|?
            'datatype': 'scaler',  # scaler vs field
            'size': 12,  # number of forces
            'output_file': '/postProcessing/forces/0/coefficient.dat',  # depends on the type of the probe/patchProbe/etc
        },
        # 'p': {
        #     'use': 'reward',  # goes into observation or rewards
        #     'type': 'probe',  # forces|probe|?
        #     'datatype': 'scaler',  # scaler vs field
        #     'size': 3,  # number of probes
        #     'output_file': '/postProcessing/patchProbes/0/p',  # depends on the type of the probe/patchProbe/etc
        # },
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
        "preprocess_cmd": foam_preprocess_cmd,
        "run_cmd": foam_run_cmd,
        "solver_full_reset": foam_full_reset,
        "rand_seed": rand_seed,
        "postprocessing_data": postprocessing_data,
        "n_parallel_env": n_parallel_env,
        "is_dummy_run": False,
    }

    # create the environment
    # env = gym.make("FoamAdapterEnv-v0")
    t0 = time.time()
    env = OpenFoamRLEnv(options)
    # good scalability regardless of the number of parallel environments
    print(f"Run time of defining OpenFoamRLEnv is {time.time()-t0} seconds")

    for epoch in range(2):  # epochs
        t01 = time.time()
        observation, _ = env.reset(return_info=True, seed=options['rand_seed'], options=options)
        print(f"Run time of defining env.reset is {time.time()-t01} seconds")

        t02 = time.time()
        counter = 0
        while True:

            action_ref = agent(env, observation)
            action_list = []
            for p_idx in range(n_parallel_env):
                # TODO: check why env seed is not set correctly. for now np.random is reproducible
                action = abs(0.000 * np.random.randn(action_ref.shape[0],))
                action_list.append(action)

            action_list = [[-0.00001, 0.00001]]

            observation, reward, done, _ = env.step(action_list)
            print('Debug data from outer loop')
            print(f"observation:")
            for obs_item in observation:
                print(obs_item)
            print(f"reward: {reward}")

            counter += 1
            if done:
                print(f"Epoch # {epoch+1}: \"done\" after {counter} steps")
                print("-------------------------------------")
                break
        print(f'Finished epoch in {time.time()-t02} seconds')
    
    print(f"Total run time is {time.time()-t0} seconds")
