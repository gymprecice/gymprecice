import gymnasium as gym
from gymprecice.core import Adapter

from os.path import join
import numpy as np
import math
from scipy import signal
import logging

from gymprecice.envs.openfoam.utils import get_interface_patches, get_patch_geometry
from gymprecice.envs.openfoam.utils import read_line
from gymprecice.utils.fileutils import open_file

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class RotatingCylinder2DEnv(Adapter):
    def __init__(self, options, idx=0) -> None:
        super().__init__(options, idx)
        self.cylinder_origin = np.array([0, 0, 0.0223788])
        self.cylinder_axis = np.array([0, 0, 1])
        self.cylinder_radius = 0.05

        self._min_omega = -0.05
        self._max_omega = 0.05
        self._n_probes = 151
        self._n_forces = 12
        self._latest_available_sim_time = 0.335

        self.action_interval = 50
        self.reward_average_time_window = 0.335

        self._previous_action = None
        self._prerun_data_required = self._latest_available_sim_time > 0.0

        self.action_space = gym.spaces.Box(
            low=self._min_omega, high=self._max_omega, shape=(1, ), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_probes,), dtype=np.float32)


        # observations and rewards are obtained from post-processing files
        self._observation_info = {
            'filed_name': 'p',
            'n_probes': self._n_probes,  # number of probes
            'file_path': f'/postProcessing/probes/{self._latest_available_sim_time}/p',
            'file_handler': None,
            'data': None  # live data for the controlled period (t > self._latest_available_sim_time)
        }
        self._reward_info = {
            'filed_name': 'forces',
            'n_forces': self._n_forces,  # number of data columns (excluding the time column)
            'Cd_column': 1,
            'Cl_column': 3,
            'file_path': f'/postProcessing/forceCoeffs/{self._latest_available_sim_time}/coefficient.dat',
            'file_handler': None,
            'prerun_file_path': '/postProcessing/forceCoeffs/0/coefficient.dat',  # cache data to prevent unnecessary run for the no control period
            'data': None  # live data for the controlled period (t > self._latest_available_sim_time)
        }

        # find openfoam solver (we have only one openfoam solver)
        openfoam_case_name = ""
        for solver_name in self._solver_list:
            if solver_name.rpartition('-')[-1].lower() == "openfoam":
                openfoam_case_name = solver_name
                break

        self._openfoam_solver_path = join(self._env_path, openfoam_case_name)

        interface_patches = get_interface_patches(join(openfoam_case_name, "system", "preciceDict"))
        actuators = []
        for patch in interface_patches:
            if patch in self._actuator_list:
                actuators.append(patch)

        self.actuator_geometric_data = get_patch_geometry(openfoam_case_name, actuators)
        actuator_coords = []

        for patch_name in self.actuator_geometric_data.keys():
            actuator_coords.append([np.delete(coord, 2) for coord in self.actuator_geometric_data[patch_name]['face_centre']])

        self._set_precice_vectices(actuator_coords)
    
    @property
    def n_probes(self):
        return self._n_probes

    @n_probes.setter
    def n_probes(self, value):
        self._n_probes = value
        self._observation_info['n_probes'] = value
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(value,), dtype=np.float32)
    
    @property
    def n_forces(self):
        return self._n_forces

    @n_forces.setter
    def n_forces(self, value):
        assert value >=3, "Number of forceCoeff columns must be greater than 2"
        self._n_forces = value
        self._reward_info['n_forces'] = value

    @property
    def min_omega(self):
        return self._min_omega

    @min_omega.setter
    def min_omega(self, value):
        self._min_omega = value
        self.action_space = gym.spaces.Box(
            low=value, high=self._max_omega, shape=(1, ), dtype=np.float32)
        
    @property
    def max_omega(self):
        return self._max_omega

    @max_omega.setter
    def max_omega(self, value):
        self._max_omega = value
        self.action_space = gym.spaces.Box(
            low=self._min_omega, high=value, shape=(1, ), dtype=np.float32)
        
    @property
    def latest_available_sim_time(self):
        return self._latest_available_sim_time

    @latest_available_sim_time.setter
    def latest_available_sim_time(self, value):
        if value == 0.0:
            value = int(value)
        self._latest_available_sim_time = value
        self._reward_info['file_path'] = f'/postProcessing/forceCoeffs/{value}/coefficient.dat'
        self._observation_info['file_path'] = f'/postProcessing/probes/{value}/p'
        self._prerun_data_required = value > 0.0

    def step(self, action):
        return self._smooth_step(action)

    def _get_action(self, action, write_var_list):
        velocity = self._action_to_patch_field(action)
        return {write_var_list[0]: velocity}

    def _get_observation(self):
        return self._probes_to_observation()

    def _get_reward(self):
        return self._forces_to_reward()

    def _close_files(self):
        # close probes and forces files
        try:
            if self._observation_info['file_handler'] is not None:
                self._observation_info['file_handler'].close()
                self._observation_info['file_handler'] = None
            if self._reward_info['file_handler'] is not None:
                self._reward_info['file_handler'].close()
                self._reward_info['file_handler'] = None
        except Exception as err:
            logger.error(f"Can't close probes/forces file")
            raise err

    def _action_to_patch_field(self, action):
        axis = self.cylinder_axis
        omega = action

        # velocity field of the actuation patches
        U = []
        for patch_name in self.actuator_geometric_data.keys():
            nf = self.actuator_geometric_data[patch_name]['face_normal']

            U_patch = (-omega) * np.cross(self.cylinder_radius * nf, axis / np.sqrt(axis.dot(axis)))
            flux = np.array([np.dot(u, n) for u, n in zip(U_patch, nf)]).reshape(-1, 1)
            U_patch = (U_patch - flux * nf)
            U.append(U_patch)
        U_profile = np.array([np.delete(item, 2) for sublist in U for item in sublist])

        return U_profile

    def _probes_to_observation(self):
        self._read_probes_from_file()

        assert self._observation_info['data'], "probes-data is empty!"
        probes_data = self._observation_info['data']

        latest_time_data = np.array(probes_data[-1][2])  # only the last timestep and remove the time and size columns

        return np.stack(latest_time_data, axis=0)

    def _forces_to_reward(self):
        self._read_forces_from_file()

        assert self._reward_info['data'], "forces-data is empty!"
        forces_data = self._reward_info['data']

        n_lookback = int(self.reward_average_time_window // self._dt) + 1

        # get the data within a time_window for computing reward
        if self._time_window == 0:
            time_bound = [0, self.reward_average_time_window]
        else:
            time_bound = [
                (self._time_window - n_lookback) * self._dt + self.reward_average_time_window,
                self._time_window * self._dt + self.reward_average_time_window
            ]

        # avoid the starting again and again from t0 by working in reverse order
        reversed_forces_data = forces_data[::-1]
        reward_data = []

        for data_line in reversed_forces_data:
            time_stamp = data_line[0]
            if time_stamp <= time_bound[0]:
                break
            reward_data.append(data_line)

        cd = np.array([[x[0], x[2][self._reward_info['Cd_column'] - 1]] for x in reward_data[::-1]])
        cl = np.array([[x[0], x[2][self._reward_info['Cl_column'] - 1]] for x in reward_data[::-1]])

        start_time_step = cd[0, 0]
        latest_time_step = cd[-1, 0]

        # average is not correct when using adaptive time-stepping
        cd_uniform = np.interp(np.linspace(start_time_step, latest_time_step, num=100, endpoint=True), cd[:, 0], cd[:, 1])
        cl_uniform = np.interp(np.linspace(start_time_step, latest_time_step, num=100, endpoint=True), cl[:, 0], cl[:, 1])
        # for constant time stepping one can filter the signals
        cd_filtered = signal.savgol_filter(cd_uniform, 49, 0)
        cl_filtered = signal.savgol_filter(cl_uniform, 49, 0)

        reward = 3.205 - np.mean(cd_filtered) - 0.2 * np.abs(np.mean(cl_filtered))
        return reward

    def _read_probes_from_file(self):
        # sequential read of a single line (last line) of probes file at each RL-Gym step
        data_path = f"{self._openfoam_solver_path}{self._observation_info['file_path']}"

        logger.debug(f'reading pressure probes from: {data_path}')

        if self._observation_info['file_handler'] is None:
            file_object = open_file(data_path)
            self._observation_info['file_handler'] = file_object
            self._observation_info['data'] = []

        new_time_stamp = True
        latest_time_stamp = self._t + self._latest_available_sim_time
        if self._observation_info['data']:
            new_time_stamp = self._observation_info['data'][-1][0] != latest_time_stamp
             
        if new_time_stamp:
            time_stamp = 0
            while not math.isclose(time_stamp, latest_time_stamp):  # read till the end of a time-window
                while True:
                    is_comment, time_stamp, n_probes, probes_data = \
                        read_line(self._observation_info['file_handler'], self._observation_info['n_probes'])
                    if not is_comment and n_probes == self._observation_info['n_probes']:
                        break
                self._observation_info['data'].append([time_stamp, n_probes, probes_data])
            assert math.isclose(time_stamp, latest_time_stamp), f"Mismatched time data: {time_stamp} vs {self._t}"

    def _read_forces_from_file(self):
        # sequential read of a single line (last line) of forces file at each RL step
        if self._prerun_data_required:
            self._reward_info['data'] = []

            data_path = f"{self._openfoam_solver_path}{self._reward_info['prerun_file_path']}"
            logger.debug(f'reading pre-run forces from: {data_path}')

            file_object = open_file(data_path)
            self._reward_info['file_handler'] = file_object

            latest_time_stamp =  self._latest_available_sim_time            

            time_stamp = 0
            while not math.isclose(time_stamp, latest_time_stamp):  # read till the end of pre-run data
                while True:
                    is_comment, time_stamp, n_forces, forces_data = \
                        read_line(self._reward_info['file_handler'], self._reward_info['n_forces'])
                    if not is_comment and n_forces == self._reward_info['n_forces']:
                        break
                self._reward_info['data'].append([time_stamp, n_forces, forces_data])
            assert math.isclose(time_stamp, latest_time_stamp), f"Mismatched time data: {time_stamp} vs {self._t}"

            self._prerun_data_required = False

            self._reward_info['file_handler'].close()

            data_path = f"{self._openfoam_solver_path}{self._reward_info['file_path']}"
            file_object = open_file(data_path)
            self._reward_info['file_handler'] = file_object

        else:
            data_path = f"{self._openfoam_solver_path}{self._reward_info['file_path']}"

            if self._reward_info['file_handler'] is None:
                file_object = open_file(data_path)
                self._reward_info['file_handler'] = file_object
                self._reward_info['data'] = []

        logger.debug(f'reading forces from: {data_path}')

        new_time_stamp = True
        latest_time_stamp = self._t + self._latest_available_sim_time
        if self._reward_info['data']:
            new_time_stamp = self._reward_info['data'][-1][0] != latest_time_stamp
            
        if new_time_stamp:
            time_stamp = 0
            while not math.isclose(time_stamp, latest_time_stamp):  # read till the end of a time-window
                while True:
                    is_comment, time_stamp, n_forces, forces_data = \
                        read_line(self._reward_info['file_handler'], self._reward_info['n_forces'])
                    if not is_comment and n_forces == self._reward_info['n_forces']:
                        break
                self._reward_info['data'].append([time_stamp, n_forces, forces_data])
            assert math.isclose(time_stamp, latest_time_stamp), f"Mismatched time data: {time_stamp} vs {self._t}"

    def _smooth_step(self, action):
        if self._previous_action is None:
            self._previous_action =  0.0 * action

        subcycle = 0
        while subcycle < self.action_interval:
            action_fraction = 1 / (self.action_interval - subcycle)
            smoothed_action = self._previous_action + action_fraction * (action - self._previous_action)

            if isinstance(smoothed_action, np.ndarray):
                next_obs, reward, terminated, truncated, info = super().step(smoothed_action)
            else:
                next_obs, reward, terminated, truncated, info = super().step(smoothed_action.cpu().numpy())

            subcycle += 1
            if terminated or truncated:
                self._previous_action = None
                break
            else:
                self._previous_action = smoothed_action
        return next_obs, reward, terminated, truncated, info
