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
    foam_src_cmd = "./foam-functions.sh"
    foam_preprocess_cmd = "preprocess"
    foam_run_cmd = "run"
    foam_preprocess_log = "log.foam_preprocess"
    foam_run_log = "log.foam_solver"

    # if True, then the preprocessing (here: blockMesh) happens per each epoch:
    foam_full_reset = False

    # reset oprions
    options = {
        "case_path": foam_case_path, "src_cmd": foam_src_cmd,
        "preprocess_cmd": foam_preprocess_cmd, "run_cmd": foam_run_cmd,
        "preprocess_log": foam_preprocess_log, "run_log": foam_run_log,
        "solver_full_reset": foam_full_reset
    }

    # create the environment
    env = gym.make("FoamAdapterEnv-v0")

    for epoch in range(10):  # epochs
        done = False
        observation = env.reset(options=options)

        while not done:
            # take an action
            action = agetn(env, observation)

            # pass the action to the environment
            observation, reward, done, _ = env.step(action)

            if done:
                print("-------------------------------------")
                print(f"Done: epoch: {epoch + 1}, reward: {reward:.1f}")
                print("-------------------------------------")
                break


if __name__ == '__main__':
    train()
