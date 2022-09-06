from copyreg import pickle
import torch
import torch.nn as nn
#from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Box
from collections import namedtuple

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SimpleAgent():
    def __init__(self, env_fn) -> None:
        # make environment, check spaces, get obs / act dims
        self.env = env_fn()
        assert isinstance(self.env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(self.env.action_space, Box), \
            "This example only works for envs with continuous action spaces."
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.shape[0]
        self.min_act = self.env.action_space.low
        self.max_act = self.env.action_space.high

    def train(self, mlp_params, train_params):
        # trainning parameters
        lr = train_params.get('lr', 1e-2)
        epochs = train_params.get('epochs', 50)
        batch_size = train_params.get('batch_size', 50)
        render = train_params.get('render', False)
        
        
        # make core of policy network
        hidden_sizes = mlp_params['hidden_sizes']
        activation = mlp_params['activation']
        output_activation = mlp_params['output_activation']
        sizes = [self.obs_dim] + hidden_sizes + [self.n_acts]
        self.mu_net = mlp(sizes, activation, output_activation)
        log_std = np.log(0.1*(self.max_act - self.min_act)) * np.ones(self.n_acts, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

        
        # make function to compute action distribution
        def get_policy(obs):
            mu = torch.from_numpy(self.max_act) * self.mu_net(obs)
            std = torch.exp(self.log_std)
            return Normal(mu, std)
        

        # make action selection function (outputs int actions, sampled from policy)
        def get_action(obs):
            act = torch.clip(get_policy(obs).sample(), self.min_act[0], self.max_act[0])
            return act

        # make loss function whose gradient, for the right data, is policy gradient
        def compute_loss(batch):
            batch_logp = []
            batch_R = []
            for trajectory in batch:
                traj_obs = []
                traj_act = []
                traj_obs.extend(map(lambda step: step.observation, trajectory.steps))
                traj_act.extend(map(lambda step: step.action, trajectory.steps))
                batch_R.append(trajectory.reward)
                obs=torch.as_tensor(traj_obs, dtype=torch.float32)
                act=torch.as_tensor(traj_act)
                batch_logp.append(get_policy(obs).log_prob(act).sum(0))
            
            logp = torch.cat(batch_logp)
            weights = torch.as_tensor(batch_R, dtype=torch.float32)
            return -(logp * weights).mean() 

        # make optimizer
        optimizer = Adam(self.mu_net.parameters(), lr=lr)

        # for training policy
        def train_one_epoch():
            Episode = namedtuple('Episode', field_names=['reward', 'steps'])
            EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

            batch = []
            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            
            episode_steps = []
            episode_reward = 0.0 # undiscounted R(tau)


            # reset episode-specific variables
            obs = self.env.reset(return_info=False)       # first obs comes from starting distribution
            done = False            # signal from environment that episode is over
            n_envs = len(obs)

            envs_episode_steps = [[] for e in range(n_envs)]
            envs_reward = np.zeros(n_envs)


            # render first episode of each epoch
            finished_rendering_this_epoch = False

            # collect experience by acting in the environment with current policy
            while True:

                # rendering
                if (not finished_rendering_this_epoch) and render:
                    self.env.render()

                # act in the environment
                act_t = get_action(torch.as_tensor(obs, dtype=torch.float32))
                act = act_t.numpy()
                next_obs, rew, done, _ = self.env.step(act)

                envs_current_step = list(map(lambda env_obs, env_act: EpisodeStep(env_obs, env_act.tolist()), obs, act))
                for i, env_step in enumerate(envs_current_step):
                    envs_episode_steps[i].append(env_step)
                envs_reward += rew
                
                if done:
                    # if episode is over, record info about episode
                    batch_rets.extend(envs_reward.tolist())
                    batch.extend(map(lambda env_reward, env_eposide: Episode(reward=env_reward, steps=env_eposide), envs_reward, envs_episode_steps))
                    envs_episode_steps = [[] for e in range(n_envs)]
                    envs_reward = np.zeros(n_envs)

                    # won't render again this epoch
                    finished_rendering_this_epoch = True

                    # end experience loop if we have enough of it
                    if len(batch) >= batch_size:
                        break
                    else:
                        next_obs, done, ep_rews = self.env.reset(return_info=False), False, []
                
                obs = next_obs

            # take a single policy gradient update step
            optimizer.zero_grad()
            batch_loss = compute_loss(batch)                   
            batch_loss.backward()
            optimizer.step()
            return batch_loss, batch_rets

        # training loop
        with open('training_result.txt', 'w') as f:
            for i in range(epochs):
                batch_loss, batch_rets = train_one_epoch()
                print('epoch: %3d \t loss: %.3f \t reward: %.3f'%
                        (i, batch_loss, np.mean(batch_rets)), file=f)