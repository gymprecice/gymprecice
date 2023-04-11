import gymnasium as gym
import precice
from precice import (action_write_initial_data,
                     action_write_iteration_checkpoint,
                     action_read_iteration_checkpoint)

from abc import abstractmethod
import math
import numpy as np
import psutil
import subprocess
import os
import logging

from gymprecice.utils.xmlutils import get_mesh_data, get_episode_end_time
from gymprecice.utils.fileutils import make_env_dir

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Adapter(gym.Env):
    """
    Gym-preCICE adapter is a generic adapter to couple OpenAI Gym RL algorithms to
    mesh-based simulation packeges supported by preCICE.
    The adapter is a base class for all RL environments providing a common OpenAI Gym
    interface for all derived environments and implements shared preCICE functionality.
    RL environments should be derived from this class, and override the abstract methods:
    "_get_action", "_get_observation", "_get_reward", "_close_files".
    """
    metadata = {}
   

    def __init__(self, options, idx) -> None:
        super().__init__()
        try:
            self._precice_cfg = options['precice']['precice_config_file_name']
            self._solver_list = options['solvers']['name']
            self._reset_script = options['solvers']['reset_script']
            self._prerun_script = options['solvers'].get('prerun_script', self._reset_script)
            self._run_script = options['solvers']['run_script']
            self._actuator_list = options['actuators']['name']
        except KeyError as err:
            logger.error(f'Invalid key {err} in options')
            raise err
        
        self._idx = idx
        self._env_dir = f'env_{self._idx}'
        self._env_path = os.path.join(os.getcwd(), self._env_dir)
        self._controller = None
        self._controller_mesh = None
        self._scalar_variables = None
        self._vector_variables = None
        self._precice_mesh_defined = False
        self._mesh_id = None
        self._vertex_ids = None
        self._read_ids = None
        self._write_ids = None
        self._vertex_coords = None
        self._read_var_list = None
        self._write_var_list = None
        self._dt = None  # solver time-step size (dictated by preCICE)
        self._interface = None  # preCICE interface
        self._time_window = None
        self._t = None  # episode time
        self._solver = None  # mesh-based numerical solver
        self._is_reset = False
        self._first_reset = True
        self._vertex_coords_np = None
        self._steps_beyond_terminated = None

        self._set_mesh_data()
        self._episode_end_time = get_episode_end_time(self._precice_cfg)
        try:
            make_env_dir(self._env_dir, self._solver_list)
        except Exception as err:
            logger.error(f"Can't create folders: {err}")
            raise err
        
    # gym methods:
    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        logger.debug(f'Reset {self._env_dir} - start')
        
        if self._first_reset is True:
            self._launch_subprocess('prerun_solvers')
            self._first_reset = False

        
        self._close_files()
        # reset mesh-based solver
        self._launch_subprocess('reset_solvers')

        # run mesh-based solver
        assert self._solver is None, 'solver_run pointer is not cleared!'
        p_process = self._launch_subprocess('run_solvers')
        assert p_process is not None, 'slover launch failed!'
        self._solver = p_process
        self._check_subprocess_exists(self._solver)

        # initiate precice interface and read single mesh data
        self._init_precice()

        self._is_reset = True
        obs = self._get_observation()

        logger.debug(f'Reset {self._env_dir} - end')
        return obs, {}

    def step(self, action):
        logger.debug(f'Step {self._env_dir} - start')
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self._is_reset, "Call reset before using step method."

        write_data = self._get_action(action, self._write_var_list)
        self._advance(write_data)  # (3) complete the previous time-window and take the next time step 
        
        obs = self._get_observation()
        reward = self._get_reward() 
        
        terminated = not self._interface.is_coupling_ongoing()

        # delete precice object upon termination (a workaround to get precice reset)
        if terminated:
            self._interface.finalize()
            del self._interface
            logger.info("preCICE finalized and its object deleted ...\n")
            # we need to check here that solver subprocess is finalized
            self._solver = self._finalize_subprocess(self._solver)
            # reset pointers
            self._interface = None
            self._solver_full_reset = False
            self._is_reset = False
        
        logger.debug(f'Step {self._env_dir} - end')
        return obs, reward, terminated, False, {}

    def close(self):
        self._finalize()

    def _set_mesh_data(self):
        scalar_variables, vector_variables, mesh_list, controller = get_mesh_data(self._precice_cfg)

        for mesh_name in mesh_list:
            if 'controller' in mesh_name.lower():
                controller['mesh_name'] = mesh_name
                break
        self._controller = controller
        self._controller_mesh = self._controller['mesh_name']

        self._scalar_variables = scalar_variables
        self._vector_variables = vector_variables
        self._read_var_list = self._controller[self._controller_mesh]['read']
        self._write_var_list = self._controller[self._controller_mesh]['write']

    def _set_precice_vectices(self, actuator_coords):
        # define the set of vertices to exchange data through precice
        assert actuator_coords, "actuator coords is empty!"
        self._vertex_coords_np = np.array([item for sublist in actuator_coords for item in sublist])
        self._precice_mesh_defined = True

    # preCICE related methods:
    def _init_precice(self):
        assert self._interface is None, "preCICE-interface re-initialisation attempt!"
        self._interface = precice.Interface("Controller", self._precice_cfg, 0, 1)

        self._time_window = 0
        self._mesh_id = {}
        self._vertex_coords = {}
        self._vertex_ids = {}

        # TODO: extend to multiple actuator meshes and/or observation mesh
        mesh_name = self._controller['mesh_name']
        mesh_id = self._interface.get_mesh_id(mesh_name)
        self._mesh_id[mesh_name] = mesh_id

        vertex_ids = self._interface.set_mesh_vertices(mesh_id, self._vertex_coords_np)
        self._vertex_ids[mesh_name] = vertex_ids
        self._vertex_coords[mesh_name] = self._vertex_coords_np

        self._dt = self._interface.initialize()  # (1) establish connection with the solver

        self._t = self._dt

        self._read_ids = {}
        self._write_ids = {}
        # precice data from a single mesh on the solver side
        mesh_name = self._controller['mesh_name']
        for read_var in self._read_var_list:
            self._read_ids[read_var] = self._interface.get_data_id(read_var, self._mesh_id[mesh_name])
        for write_var in self._write_var_list:
            self._write_ids[write_var] = self._interface.get_data_id(write_var, self._mesh_id[mesh_name])
        
        if self._interface.is_action_required(action_write_initial_data()):
            self._interface.mark_action_fulfilled(action_write_initial_data())

        self._interface.initialize_data()  # (2) start the first time-window by taking a non-controlled time-step forward

    def _advance(self, write_data):
        self._write(write_data)

        if self._interface.is_action_required(action_write_iteration_checkpoint()):
            while True:
                self._interface.mark_action_fulfilled(action_write_iteration_checkpoint())
                self._dt = self._interface.advance(self._dt)
                self._interface.mark_action_fulfilled(action_read_iteration_checkpoint())

                if (self._interface.is_time_window_complete()):
                    break
        else:
            self._dt = self._interface.advance(self._dt)

        # increase the time before reading the probes/forces for internal consistency checks
        if self._interface.is_time_window_complete():
            self._time_window += 1
        
        if self._interface.is_coupling_ongoing():
            self._t += self._dt

        # dummy advance to finalize time-window and coupling status
        if math.isclose(self._t, self._episode_end_time) and self._interface.is_coupling_ongoing():
            if self._interface.is_action_required(action_write_iteration_checkpoint()):
                while True:
                    self._interface.mark_action_fulfilled(action_write_iteration_checkpoint())
                    self._interface.advance(self._dt)
                    self._interface.mark_action_fulfilled(action_read_iteration_checkpoint())

                    if (self._interface.is_time_window_complete()):
                        break
            else:
                self._interface.advance(self._dt)
            
    def _write(self, write_data):
        for write_var in self._write_var_list:
            if write_var in self._vector_variables:
                self._interface.write_block_vector_data(
                    self._write_ids[write_var], self._vertex_ids[self._controller_mesh], write_data[write_var])
            elif write_var in self._scalar_variables:
                self._interface.write_block_scalar_data(
                    self._write_ids[write_var], self._vertex_ids[self._controller_mesh], write_data[write_var])
            else:
                raise Exception(f'Invalid variable type: {write_var}')

    def _launch_subprocess(self, cmd):
        assert cmd in ['reset_solvers', 'prerun_solvers', 'run_solvers'], \
            "Invalid shell command - supported commands: 'reset_solvers', 'prerun_solvers', and 'run_solvers'"
        subproc_env = {key: variable for key, variable in os.environ.items() if "MPI" not in key}
        if cmd == 'reset_solvers':
            for solver in self._solver_list:
                try:
                    completed_process = subprocess.run([f"./{self._reset_script}"], shell=True, env=subproc_env, cwd=f"{self._env_dir}/{solver}")
                except Exception as err:
                    logger.error(f'Failed to run {cmd} - {self._reset_script} from the folder f"{self._env_dir}/{solver}"')
                    raise err

            if completed_process.returncode != 0:
                raise Exception(f"Subprocess was not successful - {completed_process}")
           
            return None

        elif cmd == 'prerun_solvers':
            for solver in self._solver_list:
                try:
                    completed_process = subprocess.run([f"./{self._prerun_script}"], shell=True, env=subproc_env, cwd=f"{self._env_dir}/{solver}")
                except Exception as err:
                    logger.error(f'Failed to run {cmd} - {self._prerun_script} from the folder f"{self._env_dir}/{solver}"')
                    raise err

            if completed_process.returncode != 0:
                raise Exception(f"Subprocess was not successful - {completed_process}")
            
            return None

        elif cmd == 'run_solvers':
            subproc = []
            for solver in self._solver_list:
                subproc.append(subprocess.Popen([f"./{self._run_script}"], shell=True, env=subproc_env, cwd=f"{self._env_dir}/{solver}"))
            return subproc

    def _check_subprocess_exists(self, subproc_list):
        for subproc, solver in zip(subproc_list, self._solver_list):
            # check if the spawning process exists
            if not psutil.pid_exists(subproc.pid):
                raise Exception(f'Failed subprocess - f"{self._env_dir}/{solver}" -> {subproc.args[0]}')

    def _finalize_subprocess(self, subproc_list):
        for subproc, solver in zip(subproc_list, self._solver_list):
            if subproc and psutil.pid_exists(subproc.pid):
                if psutil.Process(subproc.pid).status() != psutil.STATUS_ZOMBIE:
                    logger.info("Subprocess status is not zombie - waiting to finish ...")
                    exit_signal = subproc.wait()
                else:
                    logger.info("Subprocess status is zombie - cleaning up ...")
                    exit_signal = subproc.poll()
                # check the subprocess exit signal
                if exit_signal != 0:
                    raise Exception(f'Subprocess failed to complete its shell command - f"{self._env_dir}/{solver}" -> {subproc.args[0]}')
                logger.info(f'Subprocess successfully completed its shell command: f"{self._env_dir}/{solver}" -> {subproc.args[0]}')
        return None

    def __del__(self):
        # close all the open files
        self._close_files()
        if self._interface is not None:
            try:
                self._dummy_episode()
            except Exception as err:
                logger.error(f'Unsuccessful termination attempt - {err}')
                raise err

    def _dummy_episode(self):
        # advance with actions equal to zero till the coupling finish and finalize
        dummy_action = 0.0 * self.action_space.sample()
        done = False
        while not done:
            _, _, done, _, _ = self.step(dummy_action)

    def _finalize(self):
        self.__del__()

    @abstractmethod
    def _get_action(self, action, write_var_list):
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError

    @abstractmethod
    def _close_files(self):
        pass
