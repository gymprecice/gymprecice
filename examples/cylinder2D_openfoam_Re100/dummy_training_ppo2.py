import argparse

import copy
import gym
from OpenFoamRLEnv import OpenFoamRLEnv
from utils import fix_randseeds
import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from distutils.util import strtobool


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = np.prod(env.action_space.shape)
        self.n_obs = np.prod(env.observation_space.shape)
        # TODO: extract these from the env.spaces
        self.action_min = torch.from_numpy(env.action_space.low)
        self.action_max = torch.from_numpy(env.action_space.high)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.n_actions), std=0.1),
            nn.Tanh()
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.n_actions))

    def get_value(self, x):
        x = x.reshape(-1, self.n_obs)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.reshape(-1, self.n_obs)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # scale both the mean and std
        action_mean = action_mean * (self.action_max - self.action_min) / 2
        action_std = action_std * (self.action_max - self.action_min) / 10
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            # action = torch.clip(action, min=self.action_min, max=self.action_max)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=80,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=1e-2,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


if __name__ == '__main__':
    rand_seed = 12345
    fix_randseeds(rand_seed)

    # shell options to run the solver (this can/should be placed in a
    # separate python script)
    foam_case_path = "env01_3"
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
    foam_prerun_time = 0.335

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
    foam_prerun_cmd = f" && {foam_prerun_cmd} {foam_prerun_time} > {foam_prerun_log} 2>&1"

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
        "n_parallel_env": n_parallel_env,
        "is_dummy_run": False,
        "prerun": True,
        "is_dummy_run": False
    }

    # create the environment
    # env = gym.make("FoamAdapterEnv-v0")
    t0 = time.time()
    envs = OpenFoamRLEnv(options)
    # good scalability regardless of the number of parallel environments
    print(f"Run time of defining OpenFoamRLEnv is {time.time()-t0} seconds")

    obs_dim = np.prod(envs.observation_space.shape)
    n_acts = np.prod(envs.action_space.shape)
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args = parse_args()
    agent = Agent(envs)
    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    # ALGO Logic: Storage setup --> (timesteps, num_env, n_obs) if obs is 2d then the following will be 4 dimesional
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    num_updates = 100  # args.total_timesteps // args.batch_size

    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    prev_action = None
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            if prev_action is None:
                prev_action = 0 * action

            subcycle_counter = 0
            subcycle_max = 50  # in precice config set the time step to 0.025 / 50 = 5e-4
            # subcycle_max = 25  # in precice config set the time step to 1e-3

            # little bit inefficient communication modes but lets try
            while subcycle_counter < subcycle_max:
                smoothing_fraction = (subcycle_counter / subcycle_max)
                smoothed_action = (1 - smoothing_fraction) * prev_action + smoothing_fraction * action
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(smoothed_action.cpu().numpy())
                print(f'PPO will took the following action {action} vs previous action {prev_action} at subcycle {subcycle_counter} out of {subcycle_max}, reward {reward}')
                subcycle_counter += 1

            prev_action = action

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor([done]).to(device=device)  # TODO how to fix this why do I need to put it in a list

        prev_action = 0 * action
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print(f"approx_kl is violated break at update_epochs {epoch}")
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"update: {update}, epoch: {epoch}")
        print(f"losses/value_loss {v_loss.item()}, losses/policy_loss {pg_loss.item()}")
        print(f"losses/explained_variance {explained_var}, losses/clipfrac {np.mean(clipfracs)}")
        print(f"losses/entropy {entropy_loss.item()}, losses/approx_kl {approx_kl.item()}")
        print(f"=== returns {torch.mean(rewards)}")
