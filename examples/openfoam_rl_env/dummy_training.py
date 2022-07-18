import gym
from OpenFoamRLEnvSM import OpenFoamRLEnv
from utils import fix_randseeds
import numpy as np


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
    foam_case_path = "./fluid-openfoam"
    foam_src_cmd = ". ../foam-functions.sh"
    foam_preprocess_cmd = "preprocessfoam"
    foam_run_cmd = "runfoam"
    foam_preprocess_log = "foam_preprocess.log"
    foam_run_log = "foam_run.log"

    # if True, then the preprocessing (here: blockMesh) happens per each epoch:
    foam_full_reset = False

    foam_preprocess_cmd = f"{foam_src_cmd} && {foam_preprocess_cmd} > {foam_preprocess_log} 2>&1"
    foam_run_cmd = f"{foam_src_cmd} && {foam_run_cmd} > {foam_run_log} 2>&1"

    # foam_run_cmd = f"sh ../foam-run.sh > {foam_run_log} 2>&1"
    # foam_preprocess_cmd = f"sh ../foam-preprocess.sh > {foam_preprocess_log} 2>&1"

    # reset oprions
    n_parallel_env = 4

    options = {
        "precice_cfg": "precice-config.xml",
        "case_path": foam_case_path,
        "preprocess_cmd": foam_preprocess_cmd,
        "run_cmd": foam_run_cmd,
        "solver_full_reset": foam_full_reset,
        "rand_seed": rand_seed,
        # "n_parallel": n_parallel_env,
    }

    # create the environment
    # env = gym.make("FoamAdapterEnv-v0")

    env = OpenFoamRLEnv(options)

    for epoch in range(2):  # epochs
        observation, _ = env.reset(return_info=True, seed=options['rand_seed'], options=options)
        counter = 0
        while True:
            action = agent(env, observation)
            # TODO: check why env seed is not set correctly. for now np.random is reproducable
            action = np.random.randn(action.shape[0])
            observation, reward, done, _ = env.step(action)
            counter += 1
            if done:
                print(f"Epoch # {epoch+1}: \"done\" after {counter} steps")
                print("-------------------------------------")
                break
