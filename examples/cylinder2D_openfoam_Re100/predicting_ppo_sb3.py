from stable_baselines3 import PPO
from utils import fix_randseeds
from OpenFoamRLEnv import OpenFoamRLEnv
import torch

def smooth_step(env, action, prev_action, subcycle_max=50):
    # smooth the action and step 
    subcycle_counter = 0
    
    new_obs = rewards = dones = infos = None

    # little bit inefficient communication modes but lets try
    while subcycle_counter < subcycle_max:
        smoothing_fraction = (subcycle_counter / subcycle_max)
        smoothed_action = (1 - smoothing_fraction) * prev_action + smoothing_fraction * action
        
        # TRY NOT TO MODIFY: execute the game and log data.
        new_obs, rewards, dones, infos = env.step(smoothed_action)
        subcycle_counter += 1
    return new_obs, rewards, dones, infos
        
if __name__ == '__main__':
    rand_seed = 12345
    fix_randseeds(rand_seed)

    # shell options to run the solver (this can/should be placed in a
    # separate python script)
    foam_case_path = "cylinder2D-unstructured-mesh"
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
        'p': {
            'use': 'observation',  # goes into observation or rewards
            'type': 'probe',  # forces|probe|?
            'datatype': 'scaler',  # scaler vs field
            'size': 11,  # number of probes
            'output_file': '/postProcessing/probes/0/p',  # depends on the type of the probe/patchProbe/etc
        }
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
        "prerun": True,
        "prerun_available": True,
        "prerun_time": 0.335
    }

    env = OpenFoamRLEnv(options)
    
    # load model
    model = PPO.load("../model", env=env, device="cpu", print_system_info=True)
    
    obs = env.reset()
    done = False
    prev_action = 0
    while not done:
        action, _ = model.predict(torch.as_tensor(obs).to(model.device).float())
        obs, reward, done, info = smooth_step(env, action, prev_action)
        prev_action = action
