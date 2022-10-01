from typing import Dict, List, Optional, Tuple, Type, Union, NamedTuple
from stable_baselines3.common.vec_env import VecEnv
import collections
import gym
import torch
from torch import nn
from utils import fix_randseeds
from OpenFoamRLEnv import OpenFoamRLEnv
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Generator
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import BaseBuffer

from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

from stable_baselines3.common.vec_env import VecNormalize

from collections import deque
from gym import spaces

from gym.spaces import Discrete, Box

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

from torch.nn import functional as F

from torch.distributions.normal import Normal
from stable_baselines3.common.type_aliases import Schedule


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env, use_relative_action=False):
        super().__init__()
        self.n_actions = np.prod(env.action_space.shape)
        self.n_obs = np.prod(env.observation_space.shape)
        self.action_min = torch.from_numpy(np.copy(env.action_space.low))
        self.action_max = torch.from_numpy(np.copy(env.action_space.high))

        if use_relative_action:
            self.action_min /= 3
            self.action_max /= 3

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.n_obs, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.n_obs, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, self.n_actions), std=0.1),
            nn.Tanh()
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.n_actions))

    def get_value(self, x):
        x = x.reshape(-1, self.n_obs)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.reshape(-1, self.n_obs)
        mu = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(mu)
        action_std = torch.exp(action_logstd)

        # scale both the mean and std
        action_mean = mu * (self.action_max - self.action_min) / 2
        action_std = action_std * (self.action_max - self.action_min) / 10
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), mu


class ObservationRewardWrapper2(gym.Wrapper):
    """This wrapper will augment the observation (aka env.state) with the current action"""

    def __init__(self, env, use_relative_action, deque_size: int = 50):
        super().__init__(env)
        self.env = env
        self.use_relative_action = use_relative_action
        self.num_envs = getattr(env, "num_envs", 1)
        self.obs_queue = deque(maxlen=deque_size)

        # for relative action, observations are augmented with the current action
        if self.use_relative_action:
            low = np.append(env.observation_space.low, env.observation_space.low[0])
            high = np.append(env.observation_space.high, env.observation_space.high[0])
        else:
            low = env.observation_space.low
            high = env.observation_space.high

        # stacking is inspired by what is done
        # https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py
        low = np.repeat(low, 2, axis=0)
        high = np.repeat(high, 2, axis=0)

        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        # for relative action, observations are augmented with the current action
        if self.use_relative_action:
            if self.num_envs > 1:
                action_ = np.array(action).reshape(self.num_envs, -1)
                wrapped_observations = np.concatenate((observations, action_), axis=1)
            else:
                action_ = np.array(action).flatten()
                wrapped_observations = np.concatenate((observations, action_), axis=0)
        else:
            wrapped_observations = observations

        # return wrapped_observations, rewards, dones, infos

        self.obs_queue.append(wrapped_observations)
        if self.num_envs > 1:
            return np.concatenate((self.obs_queue[0], self.obs_queue[-1]), axis=1), rewards, dones, infos
        else:
            return np.concatenate((self.obs_queue[0], self.obs_queue[-1]), axis=0), rewards, dones, infos

    def reset(self, **kwargs):
        """Resets the environment and add fake action."""
        observations = super().reset(**kwargs)
        fake_action = self.num_envs * [0.0 * self.env.action_space.sample()]

        # for relative action, observations are augmented with the current action
        if self.use_relative_action:
            if self.num_envs > 1:
                action_ = np.array(fake_action).reshape(self.num_envs, -1)
                wrapped_observations = np.concatenate((observations, action_), axis=1)
            else:
                action_ = np.array(fake_action).flatten()
                # print(observations.shape, action_.shape)
                wrapped_observations = np.concatenate((observations, action_), axis=0)
        else:
            wrapped_observations = observations

        # return wrapped_observations
        self.obs_queue.clear()
        self.obs_queue.append(wrapped_observations)
        if self.num_envs > 1:
            return np.concatenate((self.obs_queue[0], self.obs_queue[-1]), axis=1)
        else:
            return np.concatenate((self.obs_queue[0], self.obs_queue[-1]), axis=0)


class CustomRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    nxt_observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class CustomRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.nxt_observations, self.actions, self.rewards, self.advantages = None, None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.nxt_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def set_nxt_observation(self):
        self.nxt_observations[0:-1, :] = self.observations[1:, :]
        self.nxt_observations[-1, :] = self.observations[-1, :]

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "nxt_observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> CustomRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.nxt_observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return CustomRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CustomActorCriticPolicy(BasePolicy):
    """
    Customised policy class for our Agent(actor-critic) algorithm (has both policy and value prediction).
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        env=None,
        use_relative_action=None
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        self.env = env
        self.use_relative_action = use_relative_action
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.
        """
        raise NotImplementedError()

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = Agent(self.env, use_relative_action=self.use_relative_action)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        """
        self._build_mlp_extractor()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions ans values from actor-critic according to the current policy,
        given the observations 
        """
        features = self.extract_features(obs)
        actions, log_prob, _, values, _ = self.mlp_extractor.get_action_and_value(features)

        return actions, values, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        """
        actions, _, _, _, _ = self.mlp_extractor.get_action_and_value(observation)
        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        """
        features = self.extract_features(obs)
        _, log_prob, entropy, values, mu = self.mlp_extractor.get_action_and_value(features, actions)        
        return values, log_prob, entropy, mu

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        """
        features = self.extract_features(obs)
        values = self.mlp_extractor.get_value(features)
        return values


class CustomPPO(PPO):
    def __init__(
        self, policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 5e-4,
        n_steps: int = 80,
        batch_size: int = 10,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        use_relative_action = False,
        use_caps_loss = False,
        caps_lambda = 0.0
    ):
        super().__init__(
                policy,
                env,
                learning_rate,
                n_steps,
                batch_size,
                n_epochs,
                gamma,
                gae_lambda,
                clip_range,
                clip_range_vf,
                normalize_advantage,
                ent_coef,
                vf_coef,
                max_grad_norm,
                use_sde,
                sde_sample_freq,
                target_kl,
                tensorboard_log,
                create_eval_env,
                policy_kwargs,
                verbose,
                seed,
                device,
                _init_setup_model
            )
        self.use_relative_action = use_relative_action
        self.use_caps_loss = use_caps_loss
        self.caps_lambda = caps_lambda
    
    def _setup_model(self) -> None:
        super()._setup_model()
        
        # delete the default buffer and set it as our CustomRolloutBuffer
        del self.rollout_buffer
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else CustomRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()


        action_t_1 = None
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            
            # get the action
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # smooth the action and step 
            subcycle_counter = 0
            subcycle_max = 50
            exp_smoothing_factor = 0.1
            
            prev_actions = new_obs = rewards = dones = infos = None
            # avoid carrying prev_actions across episode
            if self._last_episode_starts[0]: # TODO: valid only if all n_parallel-envs reset at the same time
                prev_actions = np.clip(rollout_buffer.actions[-1], self.action_space.low, self.action_space.high)
            else:
                prev_actions = np.clip(rollout_buffer.actions[n_steps-1], self.action_space.low, self.action_space.high)

            
            
            agent_current_action = clipped_actions
            agent_prev_action = prev_actions

            if action_t_1 is None:
                action_t_1 = 0 * agent_current_action
            
            # little bit inefficient communication modes but lets try
            while subcycle_counter < subcycle_max:
                if self.use_relative_action:
                    action_fraction = (1 / subcycle_max)
                    action_t = action_t_1 + action_fraction * agent_current_action
                else:  # this is valid for both standard and caps method
                    # action_fraction = 1 / (subcycle_max - subcycle_counter)
                    # action_t = action_t_1 + action_fraction * (agent_current_action - action_t_1)

                    # non-linear smoothing
                    action_t = agent_current_action + (agent_prev_action - agent_current_action) * (1.0 - exp_smoothing_factor)**(subcycle_counter)
                    
                    ## linear smoothing 
                    # smoothing_fraction = (subcycle_counter / subcycle_max)
                    # action_t = (1 - smoothing_fraction) * agent_prev_action + smoothing_fraction * agent_current_action

                    ## linear smoothing - Paris et. al
                    # smoothing_fraction = 1 
                    # if subcycle_counter < 20:
                    #     smoothing_fraction = (subcycle_counter / 20)
                    # action_t = (1 - smoothing_fraction) * agent_prev_action + smoothing_fraction * agent_current_action

                #clipped_action_t = np.clip(action_t, self.action_space.low, self.action_space.high)
                new_obs, rewards, dones, infos = env.step(action_t)

                subcycle_counter += 1
                
                print(f'PPO will took the following action {action_t} vs agent current action {agent_current_action} at subcycle {subcycle_counter} out of {subcycle_max}, reward {rewards}')
                
                # break the subcycle if episode ends
                if dones[0]:
                    action_t_1 = None
                    break
                else:
                    action_t_1 = action_t

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        caps_loss = 0.0

        continue_training = True

        if self.use_caps_loss:
            self.rollout_buffer.set_nxt_observation()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, mu = self.policy.evaluate_actions(rollout_data.observations, actions)
                _, _, _, nxt_mu = self.policy.evaluate_actions(rollout_data.nxt_observations)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if self.use_caps_loss:
                    caps_loss = F.mse_loss(mu, nxt_mu)
                    loss += self.caps_lambda * caps_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/caps_loss", caps_loss.detach().numpy())
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


if __name__ == '__main__':
    use_caps_loss = None
    use_relative_action = None
    caps_lambda = None
    
    run_method = 'relative_update' #'caps_loss' #'standard' # 'caps_loss' #'relative_update' # 'standard'   
    
    if run_method == 'standard':
        use_caps_loss = False
        use_relative_action = False
    elif run_method == 'caps_loss':
        use_caps_loss = True
        use_relative_action = False
        caps_lambda = 3.0
    elif run_method == 'relative_update':
        use_caps_loss = False
        use_relative_action = True
    else:
        raise NotImplementedError()



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
            'size': 151,  # number of probes
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

    # create the environment
    env = OpenFoamRLEnv(options)
    
    env = ObservationRewardWrapper2(env, use_relative_action, deque_size=50)

    model = CustomPPO(CustomActorCriticPolicy, env, policy_kwargs={'env':env, 'use_relative_action':use_relative_action}, device="cpu", verbose=1,
                        use_relative_action=use_relative_action, use_caps_loss=True, caps_lambda=1.0)

    num_updates = 1000
    buffer_size = model.env.num_envs * model.n_steps
    total_timesteps = int(num_updates * buffer_size)

    # train the model
    model.learn(total_timesteps)

    # release _thread.locks 
    env.finalize()

    # save the model
    model.save("model")