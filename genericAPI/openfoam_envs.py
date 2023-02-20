from gymprecice import Adapter
from gym import spaces
from os.path import join
from os import getcwd

import numpy as np
import math
from scipy import signal

from utils.oldutil import  open_file, robust_readline
from utils.foamutil import _get_actuation_patches, _get_patch_geometry, _make_env_dir


class JetCylinder2DEnv(Adapter):

    def __init__(self, options, idx=0) -> None:
        
        self.case_dir = options['dir_structure']['case_name']
        self.idx = idx
        self.env_dir = f'env_{self.idx}'
        self._env_path = join(getcwd(), self.env_dir)
        _make_env_dir(self.env_dir, self.case_dir)

        self.cylinder_origin = np.array([0, 0, 0.0223788])
        self.cylinder_radius = 0.05
        self.jet_angle = [90, 270]
        self.jet_width = [10, 10]
        self.max_jet_rate = 2.5e-4
        self.min_jet_rate = -2.5e-4
        self.n_probes = 151
        self.shedding_vortex_dt = 0.335 # for Re=100
        self._previous_action = None
        self._steps_per_action = 50
        
        self.action_space = spaces.Box(
            low=self.min_jet_rate, high=self.max_jet_rate, shape=(1, ), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_probes,), dtype=np.float32)
        
        options['dir_structure']['env_name'] = self.env_dir
       
        # observations and rewards are obtained from post-processing files
        self._observation_info = {
            'filed_name': 'p',
            'n_probes': self.n_probes,  # number of probes 
            'dumped_file_path': '/postProcessing/probes/0/p',
            'file_handler': None,
            'data': None
        }
        self._reward_info = {
            'filed_name': 'forces',
            'n_forces': 12,  # number of data columns (excluding the time column)
            'Cd_column': 1,
            'Cl_column': 3,
            'dumped_file_path': '/postProcessing/forces/0/coefficient.dat',
            'file_handler': None,
            'data': None
        }

        # config
        self.actuators = _get_actuation_patches(join(self.case_dir, "system", "preciceDict"))
        self.actuator_geometric_data = _get_patch_geometry(self.case_dir, self.actuators)
        actuator_coords = []
       
        for patch_name in self.actuator_geometric_data.keys():
            actuator_coords.append(self.actuator_geometric_data[patch_name]['Cf'])
        
        options['actuator_geometry'] = {'coords': actuator_coords}

        super(JetCylinder2DEnv, self).__init__(options)
    
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
        except Exception as e:
            raise Exception(f'Error: could not close probes/forces file: {e}')   
  
    def _action_to_patch_field(self, action):
        theta0 = [np.radians(x) for x in self.jet_angle]
        w = [np.radians(x) for x in self.jet_width]
        origin = self.cylinder_origin
        radius = self.cylinder_radius
        patch_flow_rate = [-action, action]

        # velocity field of the actuation patches
        U = []

        for idx, patch_name in enumerate(self.actuator_geometric_data.keys()):
            patch_ctr = np.array([radius * np.cos(theta0[idx]), radius * np.sin(theta0[idx]), origin[2]])
            magSf = self.actuator_geometric_data[patch_name]['magSf']
            Sf = self.actuator_geometric_data[patch_name]['Sf']
            Cf = self.actuator_geometric_data[patch_name]['Cf']
            nf = self.actuator_geometric_data[patch_name]['nf']
            w_patch = w[idx]

            # convert volumetric flow rate to a sinusoidal profile on the interface
            avg_U = patch_flow_rate[idx] / np.sum(magSf)
            d = (patch_ctr - origin) / (np.sqrt((patch_ctr - origin).dot((patch_ctr - origin))))

            U_patch = np.zeros((Cf.shape[0], 3))

            # estimate flow rate based on the sinusoidal profile
            Q_calc = 0
            for i, c in enumerate(Cf):
                r = (c - origin) / (np.sqrt((c - origin).dot((c - origin))))
                theta = np.arccos(np.dot(r, d))
                U_patch[i] = avg_U * np.pi / 2 * np.cos(np.pi / w_patch * theta) * nf[i]
                Q_calc += U_patch[i].dot(Sf[i])

            # correct velocity profile to enforce mass conservation
            Q_err = patch_flow_rate[idx] - Q_calc
            U_err = Q_err / np.sum(magSf) * nf
            U_patch += U_err

            # return the velocity profile
            Q_final = 0
            for i, Uf in enumerate(U_patch):
                Q_final += Uf.dot(Sf[i])

            if np.isclose(Q_final, patch_flow_rate[idx]):
                U.append(U_patch)
            else:
                raise Exception('estimated velocity profile violates mass conservation')

        U_profile = np.array([item for sublist in U for item in sublist])

        return U_profile

    def _probes_to_observation(self):
        self._read_probes_from_file()        

        assert self._observation_info['data'], "Error: probes data is empty!" 
        probes_data = self._observation_info['data']
        
        latest_time_data = np.array(probes_data[-1][2])  # only the last timestep and remove the time and size columns

        return np.stack(latest_time_data, axis=0)

    def _forces_to_reward(self):
        self._read_forces_from_file()

        assert self._reward_info['data'], "Error: forces data is empty!" 
        forces_data = self._reward_info['data']

        n_lookback = int(self.shedding_vortex_dt // self._dt) + 1

        # get the data within a time_window for computing reward
        if self._time_window == 0:
            time_bound = [0, self.shedding_vortex_dt]
        else:
            time_bound = [
                (self._time_window - n_lookback) * self._dt + self.shedding_vortex_dt,
                self._time_window * self._dt + self.shedding_vortex_dt
            ]

        # avoid the starting again and again from t0 by working in reverse order
        reversed_forces_data = forces_data[::-1]
        reward_data = []

        if (reversed_forces_data[0][0] <= time_bound[1]):
            reward_data = forces_data
        else:
            for data_line in reversed_forces_data:
                time_stamp = data_line[0]
                if time_stamp <= time_bound[0]:
                    break
                reward_data.append(data_line)
        

        cd = np.array([[x[0], x[2][self._reward_info['Cd_column'] - 1]] for x in reward_data])
        cl = np.array([[x[0], x[2][self._reward_info['Cl_column'] - 1]] for x in reward_data])

        start_time_step = cd[0, 0]
        latest_time_step = cd[-1, 0]
        if latest_time_step > self.shedding_vortex_dt:
            start_time_step = latest_time_step - self.shedding_vortex_dt
        
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
        data_path = f"{self._env_path}{self._observation_info['dumped_file_path']}"
        print(f'reading pressure probes from: {data_path}')

        if self._observation_info['file_handler'] is None:
            file_object = open_file(data_path)
            self._observation_info['file_handler'] = file_object
            self._observation_info['data'] = []

        probes_time_stamp = 0
        while not math.isclose(probes_time_stamp, self._t):  # read till the end of a time-window         
            while True:
                is_comment, probes_time_stamp, n_probes, probes_data = \
                    robust_readline(self._observation_info['file_handler'], self._observation_info['n_probes'], sleep_time=0.01)
                if not is_comment and n_probes == self._observation_info['n_probes']:
                    break
            self._observation_info['data'].append([probes_time_stamp, n_probes, probes_data])
        assert math.isclose(probes_time_stamp, self._t), f"Error: mismatched time data: {probes_time_stamp} vs {self._t}"  

    def _read_forces_from_file(self):
        # sequential read of a single line (last line) of forces file at each RL step
        data_path = f"{self._env_path}{self._reward_info['dumped_file_path']}"
        print(f'reading forces from: {data_path}')

        if self._reward_info['file_handler'] is None:
            file_object = open_file(data_path)
            self._reward_info['file_handler'] = file_object
            self._reward_info['data'] = []

        forces_time_stamp = 0
        while not math.isclose(forces_time_stamp, self._t):  # read till the end of a time-window         
            while True:
                is_comment, forces_time_stamp, n_forces, forces_data = \
                    robust_readline(self._reward_info['file_handler'], self._reward_info['n_forces'], sleep_time=0.01)
                if not is_comment and n_forces == self._reward_info['n_forces']:
                    break
            self._reward_info['data'].append([forces_time_stamp, n_forces, forces_data])
        assert math.isclose(forces_time_stamp, self._t), f"Error: mismatched time data: {forces_time_stamp} vs {self._t}"  

    def _smooth_step(self, action):
        if self._previous_action is None:
            self._previous_action = 0 * action

        subcycle = 0
        while subcycle < self._steps_per_action:
            action_fraction = 1 / (self._steps_per_action - subcycle)
            smoothed_action = self._previous_action + action_fraction * (action - self._previous_action)
            
            if isinstance(smoothed_action ,np.ndarray):
                next_obs, reward, done, info =  super(JetCylinder2DEnv, self).step(smoothed_action)
            else:
                next_obs, reward, done, info =  super(JetCylinder2DEnv, self).step(smoothed_action.cpu().numpy())
            
            subcycle += 1
        
            if done:
                self._previous_action = None
                break
            else:
                self._previous_action = smoothed_action
        return next_obs, reward, done, info
    
    
class RotatingCylinder2DEnv(Adapter):

    def __init__(self, options, idx=0) -> None:
        
        self.case_dir = options['dir_structure']['case_name']
        self.idx = idx
        self.env_dir = f'env_{self.idx}'
        self._env_path = join(getcwd(), self.env_dir)
        _make_env_dir(self.env_dir, self.case_dir)

        self.cylinder_origin = np.array([0, 0, 0.0223788])
        self.cylinder_axis = np.array([0, 0, 1])
        self.cylinder_radius = 0.05
       
        self.min_omega = -0.05 
        self.max_omega = 0.05
        
        self.n_probes = 151
        self.shedding_vortex_dt = 0.335 # for Re=100
        self._previous_action = None
        self._steps_per_action = 50
        
        self.action_space = spaces.Box(
            low=self.min_omega, high=self.max_omega, shape=(1, ), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_probes,), dtype=np.float32)
        
        options['dir_structure']['env_name'] = self.env_dir
       
        # observations and rewards are obtained from post-processing files
        self._observation_info = {
            'filed_name': 'p',
            'n_probes': self.n_probes,  # number of probes 
            'dumped_file_path': '/postProcessing/probes/0/p',
            'file_handler': None,
            'data': None
        }
        self._reward_info = {
            'filed_name': 'forces',
            'n_forces': 12,  # number of data columns (excluding the time column)
            'Cd_column': 1,
            'Cl_column': 3,
            'dumped_file_path': '/postProcessing/forces/0/coefficient.dat',
            'file_handler': None,
            'data': None
        }

        # config
        self.actuators = _get_actuation_patches(join(self.case_dir, "system", "preciceDict"))
        self.actuator_geometric_data = _get_patch_geometry(self.case_dir, self.actuators)
        actuator_coords = []
       
        for patch_name in self.actuator_geometric_data.keys():
            actuator_coords.append(self.actuator_geometric_data[patch_name]['Cf'])
        
        options['actuator_geometry'] = {'coords': actuator_coords}

        super(RotatingCylinder2DEnv, self).__init__(options)
    
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
        except Exception as e:
            raise Exception(f'Error: could not close probes/forces file: {e}')   
  
    def _action_to_patch_field(self, action):
        origin = self.cylinder_origin
        axis = self.cylinder_axis
        omega = action 

        # velocity field of the actuation patches
        U = []

        for idx, patch_name in enumerate(self.actuator_geometric_data.keys()):
            Cf = self.actuator_geometric_data[patch_name]['Cf']
            nf = self.actuator_geometric_data[patch_name]['nf']
            U_patch = np.zeros((Cf.shape[0], 3))
            for i, c in enumerate(Cf):
                U_patch[i] = (-omega) * np.cross(c - origin,  axis / np.sqrt(axis.dot(axis)))
                U_patch[i] = (U_patch[i] - nf[i] * (np.dot(nf[i], U_patch[i])))
            U.append(U_patch)
           
        U_profile = np.array([item for sublist in U for item in sublist])

        return U_profile

    def _probes_to_observation(self):
        self._read_probes_from_file()        

        assert self._observation_info['data'], "Error: probes data is empty!" 
        probes_data = self._observation_info['data']
        
        latest_time_data = np.array(probes_data[-1][2])  # only the last timestep and remove the time and size columns

        return np.stack(latest_time_data, axis=0)

    def _forces_to_reward(self):
        self._read_forces_from_file()

        assert self._reward_info['data'], "Error: forces data is empty!" 
        forces_data = self._reward_info['data']

        n_lookback = int(self.shedding_vortex_dt // self._dt) + 1

        # get the data within a time_window for computing reward
        if self._time_window == 0:
            time_bound = [0, self.shedding_vortex_dt]
        else:
            time_bound = [
                (self._time_window - n_lookback) * self._dt + self.shedding_vortex_dt,
                self._time_window * self._dt + self.shedding_vortex_dt
            ]

        # avoid the starting again and again from t0 by working in reverse order
        reversed_forces_data = forces_data[::-1]
        reward_data = []

        if (reversed_forces_data[0][0] <= time_bound[1]):
            reward_data = forces_data
        else:
            for data_line in reversed_forces_data:
                time_stamp = data_line[0]
                if time_stamp <= time_bound[0]:
                    break
                reward_data.append(data_line)
        

        cd = np.array([[x[0], x[2][self._reward_info['Cd_column'] - 1]] for x in reward_data])
        cl = np.array([[x[0], x[2][self._reward_info['Cl_column'] - 1]] for x in reward_data])

        start_time_step = cd[0, 0]
        latest_time_step = cd[-1, 0]
        if latest_time_step > self.shedding_vortex_dt:
            start_time_step = latest_time_step - self.shedding_vortex_dt
        
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
        data_path = f"{self._env_path}{self._observation_info['dumped_file_path']}"
        print(f'reading pressure probes from: {data_path}')

        if self._observation_info['file_handler'] is None:
            file_object = open_file(data_path)
            self._observation_info['file_handler'] = file_object
            self._observation_info['data'] = []

        probes_time_stamp = 0
        while not math.isclose(probes_time_stamp, self._t):  # read till the end of a time-window         
            while True:
                is_comment, probes_time_stamp, n_probes, probes_data = \
                    robust_readline(self._observation_info['file_handler'], self._observation_info['n_probes'], sleep_time=0.01)
                if not is_comment and n_probes == self._observation_info['n_probes']:
                    break
            self._observation_info['data'].append([probes_time_stamp, n_probes, probes_data])
        assert math.isclose(probes_time_stamp, self._t), f"Error: mismatched time data: {probes_time_stamp} vs {self._t}"  

    def _read_forces_from_file(self):
        # sequential read of a single line (last line) of forces file at each RL step
        data_path = f"{self._env_path}{self._reward_info['dumped_file_path']}"
        print(f'reading forces from: {data_path}')

        if self._reward_info['file_handler'] is None:
            file_object = open_file(data_path)
            self._reward_info['file_handler'] = file_object
            self._reward_info['data'] = []

        forces_time_stamp = 0
        while not math.isclose(forces_time_stamp, self._t):  # read till the end of a time-window         
            while True:
                is_comment, forces_time_stamp, n_forces, forces_data = \
                    robust_readline(self._reward_info['file_handler'], self._reward_info['n_forces'], sleep_time=0.01)
                if not is_comment and n_forces == self._reward_info['n_forces']:
                    break
            self._reward_info['data'].append([forces_time_stamp, n_forces, forces_data])
        assert math.isclose(forces_time_stamp, self._t), f"Error: mismatched time data: {forces_time_stamp} vs {self._t}"  

    def _smooth_step(self, action):
        if self._previous_action is None:
            self._previous_action = 0 * action

        subcycle = 0
        while subcycle < self._steps_per_action:
            action_fraction = 1 / (self._steps_per_action - subcycle)
            smoothed_action = self._previous_action + action_fraction * (action - self._previous_action)
            
            if isinstance(smoothed_action ,np.ndarray):
                next_obs, reward, done, info =  super(RotatingCylinder2DEnv, self).step(smoothed_action)
            else:
                next_obs, reward, done, info =  super(RotatingCylinder2DEnv, self).step(smoothed_action.cpu().numpy())
            
            subcycle += 1
        
            if done:
                self._previous_action = None
                break
            else:
                self._previous_action = smoothed_action
        return next_obs, reward, done, info 