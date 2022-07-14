import gym
from gymPreciceAdapter.envs import FoamAdapterEnv


def agetn(*argv):
    """
    dummy agent to return an action based on observation from env
    """
    env = argv[0]
    return env.action_space.sample()


def train():
    """
    dummy training function to test OF-precice coupling
    """
    # shell options to run the solver (this can/should be placed in a
    # separate python script)
    foam_case_path = "./fluid-openfoam"
    foam_src_cmd = ". ./foam-functions.sh"
    foam_preprocess_cmd = "preprocess"
    foam_run_cmd = "run"
    foam_preprocess_log = "foam_preprocess.log"
    foam_run_log = "foam_run.log"

    foam_preprocess_cmd = foam_src_cmd + " && " + foam_preprocess_cmd + " " \
        + foam_case_path + " > " + foam_case_path + "/" \
        + foam_preprocess_log + " 2>&1"
    foam_run_cmd = foam_src_cmd + " && " + foam_run_cmd + " " \
        + foam_case_path + " > " + foam_case_path + "/" \
        + foam_run_log + " 2>&1"

    # if True, then the preprocessing (here: blockMesh) happens per each epoch:
    foam_full_reset = False

    # reset oprions
    options = {
        "case_path": foam_case_path,
        "preprocess_cmd": foam_preprocess_cmd, "run_cmd": foam_run_cmd,
        "solver_full_reset": foam_full_reset
    }

    # create the environment
    env = gym.make("FoamAdapterEnv-v0")

    for epoch in range(1):  # epochs
        observation, _ = env.reset(return_info=True, options=options)
        counter = 0
        while True:
            action = agetn(env, observation)
            observation, reward, done, _ = env.step(action)
            counter += 1
            if done:
                print(f"Epoch # {epoch+1}: \"done\" after {counter} steps")
                print("-------------------------------------")
                break


if __name__ == '__main__':
    train()
