from typing import Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.vec_env import VecEnv
import collections
import gym
import torch
from torch import nn
from utils import fix_randseeds
from OpenFoamRLEnv import OpenFoamRLEnv
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)

from torch.distributions.normal import Normal
from stable_baselines3.common.type_aliases import Schedule


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = np.prod(env.action_space.shape)
        self.n_obs = np.prod(env.observation_space.shape)
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
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
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
        env=None
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
        self.mlp_extractor = Agent(self.env)

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
        actions, log_prob, _, values = self.mlp_extractor.get_action_and_value(features)

        return actions, values, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        """
        actions, _, _, _ = self.mlp_extractor.get_action_and_value(observation)
        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        """
        features = self.extract_features(obs)
        _, log_prob, entropy, values = self.mlp_extractor.get_action_and_value(features, actions)        
        return values, log_prob, entropy

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
        n_steps: int = 160,
        batch_size: int = 16,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
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
        _init_setup_model: bool = True
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
                _init_setup_model,
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
            
            prev_actions = new_obs = rewards = dones = infos = None
            # avoid carrying prev_actions across episode
            if self._last_episode_starts[0]: # TODO: valid only if all n_parallel-envs reset at the same time
                prev_actions = rollout_buffer.actions[-1]
            else:
                prev_actions = rollout_buffer.actions[n_steps-1]

            # little bit inefficient communication modes but lets try
            while subcycle_counter < subcycle_max:
                smoothing_fraction = (subcycle_counter / subcycle_max)
                smoothed_action = (1 - smoothing_fraction) * prev_actions + smoothing_fraction * clipped_actions
                
                # TRY NOT TO MODIFY: execute the game and log data.
                new_obs, rewards, dones, infos = env.step(smoothed_action)
                print(f'PPO will took the following action {smoothed_action} vs previous action {prev_actions} at subcycle {subcycle_counter} out of {subcycle_max}, reward {rewards}')
                subcycle_counter += 1
                # break the subcycle if episode ends
                if dones[0]: # TODO: valid only if all n_parallel-envs reset at the same time
                    break

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


if __name__ == '__main__':
    rand_seed = 12345
    fix_randseeds(rand_seed)

    # shell options to run the solver (this can/should be placed in a
    # separate python script)
    foam_case_path = "cylinder2D-unstructured-test"
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
        "prerun_time": 0.335
    }

    # create the environment
    env = OpenFoamRLEnv(options)

    model = CustomPPO(CustomActorCriticPolicy, env, policy_kwargs={'env':env}, device="cpu", verbose=1)

    num_updates = 100
    buffer_size = model.env.num_envs * model.n_steps
    total_timesteps = int(num_updates * buffer_size)

    model.learn(total_timesteps)