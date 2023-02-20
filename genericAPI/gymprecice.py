import gym
import precice

from abc import abstractmethod

from precice import action_write_initial_data
import numpy as np

import psutil
import subprocess

from utils.oldutil import (get_mesh_data, get_episode_end_time)


class Adapter(gym.Env):
    """
    Gym-preCICE adapter is a generic adapter to couple OpenAI Gym RL algorithms to 
    mesh-based simulation packeges supported by preCICE.
    The adapter is base class for all RL environments providing a common OpenAI Gym 
    interface for all derived environments and implements shared preCICE functionality.
    RL environments should be derived from this class, and override the abstract methods:
    "_get_action", "_get_observation", "_get_reward", "_close_files".
    """

    metadata = {}

    def __init__(self, options) -> None:
        super().__init__()
        
        self._precice_cfg = options['dir_structure']['precice_config_file_name']
        self._env_dir = options['dir_structure']['env_name']
        self._reset_script = options['shell_scripts']['reset_solver']
        self._prerun_script = options['shell_scripts']['prerun_solver']
        self._run_script = options['shell_scripts']['run_solver']

        self.actuator_coords = options['actuator_geometry']['coords']

        scaler_variables, vector_variables, mesh_list, RL_participant = get_mesh_data('', self._precice_cfg)
        
        for mesh_name in mesh_list:
            if 'rl' in mesh_name.lower():
                RL_participant['mesh_name'] = mesh_name
                break
        self._RL_participant = RL_participant
        self._RL_mesh = self._RL_participant['mesh_name']

        # scaler and vector variables should be used to define the size of action space
        self._scaler_variables = scaler_variables
        self._vector_variables = vector_variables
        self._mesh_id = None
        self._vertex_ids = None
        self._read_ids = None
        self._write_ids = None
        self._vertex_coords = None
        self._read_var_list = None
        self._write_var_list = None

        try:
            self._read_var_list = self._RL_participant[self._RL_mesh]['read']
        except Exception as e:
            # a valid situation when the mesh doesn't have any read variables
            self._read_var_list = []
        try:
            self._write_var_list = self._RL_participant[self._RL_mesh]['write']
        except Exception as e:
            # a valid situation when the mesh doesn't have any write variables
            self._write_var_list = []

        # coupling attributes:
        self._episode_end_time = float(get_episode_end_time('', self._precice_cfg))
        self._dt = None  # solver time-step size (dictated by preCICE)
        self._interface = None  # preCICE interface
        self._time_window = None
        self._t = None  # episode time

        self._solver = None  # mesh-based numerical solver

        self._launch_subprocess('reset_solver')
        self._launch_subprocess('prerun_solver')

    # gym methods:
    def reset(self, *, seed=None, return_info=False):
        super().reset(seed=seed)
        
        print(f'reset: {self._env_dir}')

        self._close_files()
        # reset mesh-based solver
        self._launch_subprocess('reset_solver')

        # run mesh-based solver
        assert self._solver is None, 'Error: solver_run pointer is not cleared!'
        p_process = self._launch_subprocess('run_solver')
        assert p_process is not None, 'Error: slover launch failed!'
        self._solver = p_process
        self._check_subprocess(self._solver)

       
       # initiate precice interface and read single mesh data
        self._init_precice()

        if self._interface.is_action_required(action_write_initial_data()):
            self._interface.mark_action_fulfilled(action_write_initial_data())

        self._interface.initialize_data()  # if initialize="True" --> push solver one time-window forward
        self._t = self._dt
        self._is_data_initialized = True

        self._is_reset = True

        obs = self._get_observation()

        print(f'End: reset: {self._env_dir}')
 
        return obs


    def step(self, action):
        print(f'step: {self._env_dir}')

        if not isinstance(action, list) and not isinstance(action, np.ndarray):
            raise Exception("Action should be either a list or numpy array")
        if isinstance(action, np.ndarray) and len(action.shape) == 2 and action.shape[0] == 1:
            action = action[0, :]

        if not self._is_reset:
            raise Exception("Call reset before interacting with the environment.")

        write_data = self._get_action(action, self._write_var_list)  

        self._advance(write_data)  # run fluid solver to complete the next time-window

        obs = self._get_observation()
        reward = self._get_reward()
        done = not self._interface.is_coupling_ongoing()

        # delete precice object upon done (a workaround to get precice reset)
        if done:
            self._interface.finalize()
            del self._interface
            print("preCICE finalized and its object deleted ...\n")
            # we need to check here that solver run is finalized
            self._solver = self._finalize_subprocess(self._solver)
            # reset pointers
            self._interface = None
            self._solver_full_reset = False
            self._is_reset = False
        
        return obs, reward, done, {}


    def close(self):
        self._finalize()


    # preCICE related methods:
    def _init_precice(self):
        assert self._interface is None, "Error: precice interface re-initialisation attempt!"
        
        self._interface = precice.Interface("RL", self._precice_cfg, 0, 1)

        self._time_window = 0
        self._mesh_id = {}
        self._vertex_coords = {}
        self._vertex_ids = {}
        
        mesh_name = self._RL_participant['mesh_name']
        mesh_id = self._interface.get_mesh_id(mesh_name)
        self._mesh_id[mesh_name] = mesh_id
        
        vertex_coords = np.array([item for sublist in self.actuator_coords for item in sublist])
        vertex_ids = self._interface.set_mesh_vertices(mesh_id, vertex_coords)
        self._vertex_ids[mesh_name] = vertex_ids
        self._vertex_coords[mesh_name] = vertex_coords

        # establish connection with the solver
        self._dt = self._interface.initialize()
    
        self._read_ids = {}
        self._write_ids = {}
        # precice data from a single mesh on the solver side
        mesh_name = self._RL_participant['mesh_name']
        for read_var in self._read_var_list:
            self._read_ids[read_var] = self._interface.get_data_id(read_var, self._mesh_id[mesh_name])
        for write_var in self._write_var_list:
            self._write_ids[write_var] = self._interface.get_data_id(write_var, self._mesh_id[mesh_name])
   
    def _advance(self, write_data):
        self._write(write_data)
        self._interface.advance(self._dt)
        # increase the time before reading the probes/forces for internal consistency checks
        if self._interface.is_time_window_complete():
            self._time_window += 1
        self._t += self._dt

        # dummy advance to finalize time-window and coupling status
        if np.isclose(self._t, self._episode_end_time):
            self._interface.advance(self._dt)

    def _write(self, write_data):
        for write_var in self._write_var_list:
            if write_var in self._vector_variables:
                self._interface.write_block_vector_data(
                    self._write_ids[write_var], self._vertex_ids[self._RL_mesh], write_data[write_var])
            elif write_var in self._scalar_variables:
                self._interface.write_block_scalar_data(
                    self._write_ids[write_var], self._vertex_ids[self._RL_mesh], write_data[write_var])
            else:
                raise Exception(f'Invalid variable type: {write_var}')   

    def _launch_subprocess(self, cmd):
        assert cmd in ['reset_solver', 'prerun_solver', 'run_solver'], \
            "Error: invalid shell command - supported commands: 'reset_solver', 'prerun_solver', and 'run_solver'"

        if cmd == 'reset_solver':
            try:
                completed_process = subprocess.run([f"./{self._reset_script}"], shell=True, cwd=self._env_dir)
            except Exception as e:
                raise Exception(f'failed to run {cmd} - {self._reset_script} from the folder {self._env_dir}')

            if completed_process.returncode != 0:
                raise Exception(f"run is not successful - {completed_process}")
            return None

        elif cmd == 'prerun_solver':
            try:
                completed_process = subprocess.run([f"./{self._prerun_script}"], shell=True, cwd=self._env_dir)
            except Exception as e:
                raise Exception(f'failed to run {cmd} - {self._prerun_script} from the folder {self._env_dir}')

            if completed_process.returncode != 0:
                raise Exception(f"run is not successful - {completed_process}")
            return None

        elif cmd == 'run_solver':
            subproc = subprocess.Popen([f"./{self._run_script}"], shell=True, cwd=self._env_dir)
            return subproc

    def _check_subprocess(self, subproc): 
        # check if the spawning process is successful
        if not psutil.pid_exists(subproc.pid):
            raise Exception(f'Error: subprocess failed to be launched - {self._env_dir} -> {subproc.args[0]}')

        # finalize the subprocess if it is terminated (normally/abnormally)
        if psutil.Process(subproc.pid).status() == psutil.STATUS_ZOMBIE:
            print(psutil.Process(subproc.pid), psutil.Process(subproc.pid).status())
            raise Exception(f'Error: subprocess failed to be launched - {self._env_dir} -> {subproc.args[0]}')

    def _finalize_subprocess(self, subproc):
        if subproc and psutil.pid_exists(subproc.pid):
            if psutil.Process(subproc.pid).status() != psutil.STATUS_ZOMBIE:
                print("subprocess status is not zombie - waiting to finish ...")
                exit_signal = subproc.wait()
            else:
                print("subprocess status is zombie - cleaning up ...")
                exit_signal = subproc.poll()
            # check the subprocess exit signal
            if exit_signal != 0:
                raise Exception(f'subprocess failed to complete its shell command - {self._env_dir} -> {subproc.args[0]}')
            print(f'subprocess successfully completed its shell command: {self._env_dir} -> {subproc.args[0]}')
        return None

    def __del__(self):
        # close all the open files
        self._close_files()
        if self._interface:
            try:
                self.dummy_episode()
            except Exception as e:
                raise Exception(f"Unsuccessful termination attempt - {e}")

    def dummy_episode(self):
        # advance with actions equal to zero till the coupling finish and finalize
        dummy_action = 0.0 * self.action_space.sample()
        done = False
        while not done:
            _, _, done, _ = self.step(dummy_action)
    
    def _finalize(self):
        self.__del__()


    @abstractmethod
    def _get_action(self, action):
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


    
