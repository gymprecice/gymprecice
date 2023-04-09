import gymnasium as gym  
from os import path, getcwd

class Adapter(gym.Env):
    """
    Mock class representing gymprecice adapter in all environment tests.
    """
    metadata = {}
    def __init__(self, options, *args) -> None:
        self._solver_list = options['solvers']['name']
        self._actuator_list = options['actuators']['name']
        self._env_dir = "env_0"
        self._env_path = path.join(getcwd(), self._env_dir)
        self._t = 0
        self._dt = 0.0005
        self._time_window = 0

    def reset(self, *args):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _set_precice_vectices(self, actuator_coords):
       raise NotImplementedError

    def _init_precice(self):
        raise NotImplementedError

    def _advance(self, write_data):
        raise NotImplementedError
 
    def _write(self, write_data):
       raise NotImplementedError

    def _launch_subprocess(self, cmd):
        raise NotImplementedError

    def _check_subprocess_exists(self, subproc_list):
       raise NotImplementedError

    def _finalize_subprocess(self, subproc_list):
        raise NotImplementedError

    def __del__(self):
        pass

    def _dummy_episode(self):
       raise NotImplementedError

    def _finalize(self):
       raise NotImplementedError

    def _get_action(self, action):
        raise NotImplementedError

    def _get_observation(self):
        raise NotImplementedError

    def _get_reward(self):
        raise NotImplementedError

    def _close_files(self):
        raise NotImplementedError
