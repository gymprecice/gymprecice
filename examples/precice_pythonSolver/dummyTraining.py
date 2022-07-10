import gym
import gymPreciceAdapter

# shell command to run the solver
cmd_solver = "python3 -u ./fluid-python/FluidSolver.py ./precice-config.xml >> ./fluid-python/log.solver 2>&1 "
options={"cmd_solver":cmd_solver}

# create the environment
env = gym.make("FluidSolverAdapterEnv-v0")

for i in range(10): #epochs
    done = False
    observation = env.reset(options=options)
    
    while not done:
        # take an action
        action = env.action_space.sample()

        # pass the action to the environment
        observation, reward, done, info = env.step(action) 

        if done:
            break
