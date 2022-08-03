# from cmath import inf
from cmath import acos
from signal import signal
import subprocess
import argparse
from tkinter.messagebox import NO
import numpy as np
import math
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint
from pathlib import Path
from datetime import datetime
import os

import gym
from gym import spaces
from mesh_parser import FoamMesh

import time
import psutil
from utils import get_cfg_data, parse_probe_lines, make_parallel_config, load_file, parallel_precice_dict
import xmltodict
import copy


class OpenFoamRLEnv(gym.Env):
    """
    RL Env via a dummy precice adapter class
    standard OpenAI gym functions (step, reset, close, render)
    internal functions starts with '_' (e.g. _make_run_folders)
    problem setup functions starts with setup_ (e.g. setup_env_obs_act)
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, options) -> None:
        super().__init__()

        self.__options = copy.deepcopy(options)

        # gym env attributes:
        self.__is_initialized = False
        self.__is_first_reset = True  # if True, gym env reset has been called at least once
        # action_ and observation_space will be set in _set_precice_data
        self.action_space = None
        self.observation_space = None
        self.run_folders = self._make_run_folders()

        scaler_variables, vector_variables, mesh_list, mesh_variables = \
            get_cfg_data('', self.__options['precice_cfg'])
        print(scaler_variables, vector_variables, mesh_list, mesh_variables)
        # select only RL-Gym meshes
        mesh_list = [x for x in mesh_list if 'rl' in x.lower()]
        self.__mesh_variables = mesh_variables
        self.__mesh_list = mesh_list

        # scaler and vector variables should be used to define the size of action space
        self.scaler_variables = scaler_variables
        self.vector_variables = vector_variables

        # coupling attributes:
        self.__precice_dt = None
        self.__interface = None  # preCICE interface
        self.__mesh_id = None
        self.__dim = None
        self.__vertex_ids = None

        # solver attributes:
        self.__solver_run = []  # physical solver
        self.__solver_full_reset = False  # if True, run foam-preprocess upon every reset

        # observations and rewards are obtained from post-processing files
        self.__probes_rewards_data = {}
        self.__postprocessing_filehandler_dict = {}
        self.__precice_read_data = {}

    def __del__(self):
        # close all the open files
        self._close_postprocessing_files()
        if self.__interface:
            try:
                print('crashing the interface using negative timestep')
                self.__interface.advance(-1)
                # another idea is to advance with random actions till the coupling finish and finalize
                # self.__interface.finalize()
            except Exception as e:
                pass
        # print('delete the interface')
        # self.__interface.finalize()

    def _make_run_folders(self):
        # 1- clean the case file
        # 2- run grid preprocessor on the original OpenFoam case files
        # 3- parse mesh data from the case folder to get the rl-grid
        # 4- duplicate xml file for parallel processing
        # 5- replicate the folders using a mix of symbolic links and modified preciceDict

        shell_cmd = self.__options.get("foam_shell_cmd", "")
        case_path = self.__options.get("case_path", "")
        clean_cmd = self.__options.get("clean_cmd", "")
        preprocess_cmd = self.__options.get("preprocess_cmd", "")

        self._launch_subprocess(shell_cmd, clean_cmd, case_path, cmd_type='clean')
        self._launch_subprocess(shell_cmd, preprocess_cmd, case_path, cmd_type='preprocess')

        # parse solver mesh data
        self.setup_mesh_data(case_path)

        # Create an empty folder for the RL_Gym to run OpenFoam
        cwd = Path.cwd()
        time_str = datetime.now().strftime('%d%m%Y_%H%M%S')
        run_folder_name = f'rl_gym_run_{time_str}'
        run_folder = cwd.joinpath(run_folder_name)
        try:
            run_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f'failed to create run folder: {e}')

        # Create a n_parallel case_folders as symbolic links
        case_path_str = self.__options['case_path']
        source_folder_str = str(cwd) + '/' + case_path_str
        run_folder_list = []

        for idx_ in range(self.__options['n_parallel_env']):
            dist_folder_str = str(run_folder) + '/' + case_path_str + f'_{idx_}'
            run_folder_list.append(dist_folder_str)
            try:
                os.system(f'cp -rs {source_folder_str} {dist_folder_str}')
            except Exception as e:
                raise Exception(f'Failed to create symbolic links to foam case files: {e}')

            precicedict_str = load_file(dist_folder_str + "/system/", 'preciceDict')
            # print(precicedict_str)
            new_string = parallel_precice_dict(precicedict_str, idx_)
            # print(new_string)
            # delete the symbolic link
            complete_filepath = f'{dist_folder_str}/system/preciceDict'

            os.system(f"rm {complete_filepath}")
            with open(complete_filepath, 'w') as file_obj:
                file_obj.write(new_string)

        # Get a new version of precice-config.xml
        parallel_tree = make_parallel_config(str(cwd) + '/', "precice-config.xml", self.__options['n_parallel_env'], run_folder_list, use_mapping=True)
        precice_config_parallel_file = str(run_folder) + "/precice-config.xml"

        with open(precice_config_parallel_file, 'w') as file_obj:
            file_obj.write(xmltodict.unparse(parallel_tree, encoding='utf-8', pretty=True))
        os.chdir(str(run_folder))
        os.system(f'cp ../foam-functions.sh .')

        self.__options['precice_cfg'] = precice_config_parallel_file
        return run_folder_list

    # gym methods:
    def reset(self, *, seed=None, return_info=True, options=None):
        super().reset(seed=seed)

        # get the solver-launch options
        shell_cmd = self.__options.get("foam_shell_cmd", "")
        case_path = self.__options.get("case_path", "")
        clean_cmd = self.__options.get("clean_cmd", "")
        softclean_cmd = self.__options.get("softclean_cmd", "")
        preprocess_cmd = self.__options.get("preprocess_cmd", "")
        run_cmd = self.__options.get("run_cmd", "")
        self.__solver_full_reset = self.__options.get("solver_full_reset", self.__solver_full_reset)

        for p_idx in range(self.__options['n_parallel_env']):
            p_case_path = case_path + f'_{p_idx}'
            if False:  # or self.__is_first_reset or self.__solver_full_reset:
                # clean open-foam case
                self._launch_subprocess(shell_cmd, clean_cmd, p_case_path, cmd_type='clean')
                # self.__solver_clean = self._finalize_subprocess(self.__solver_clean)
                # assert self.__solver_clean is None
                self._launch_subprocess(shell_cmd, preprocess_cmd, p_case_path, cmd_type='preprocess')
                # self.__solver_preprocess = self._finalize_subprocess(self.__solver_preprocess)
                # assert self.__solver_preprocess is None
            else:
                # clean the log files
                self._launch_subprocess(shell_cmd, softclean_cmd, p_case_path, cmd_type='softclean')
                # self.__solver_clean = self._finalize_subprocess(self.__solver_clean)
                # assert self.__solver_clean is None

        if len(self.__postprocessing_filehandler_dict) > 0:
            self._close_postprocessing_files()

        # run open-foam solver
        if len(self.__solver_run) > 0:
            raise Exception('solver_run pointer is not cleared -- should not reach here')

        for p_idx in range(self.__options['n_parallel_env']):
            p_case_path = case_path + f'_{p_idx}'
            if self.__options['is_dummy_run']:
                p_process = self._launch_dummy_subprocess(p_idx, p_case_path)
            else:
                p_process = self._launch_subprocess(shell_cmd, run_cmd, p_case_path, cmd_type='run')

            assert p_process is not None
            self.__solver_run.append(p_process)

        # checking spawning after n_parallel calls to avoid sleeping $n times
        time.sleep(0.5)  # single wait time for all parallel runs
        for p_idx in range(self.__options['n_parallel_env']):
            p_case_path = case_path + f'_{p_idx}'
            self._check_subprocess(self.__solver_run[p_idx], run_cmd, p_case_path)

        # initiate precice interface and read single mesh data
        self._init_precice()
        self._set_precice_data()
        self.setup_env_obs_act()

        if self.__interface.is_action_required(action_write_initial_data()):
            # what is the action for this case no action have been provided
            # TODO: what is the first action before we can do this reliably
            self.__write_data = {}
            for p_idx in range(self.__options['n_parallel_env']):
                initial_action = self.setup_initial_action(p_idx)
                conv_action = self.setup_patch_field_to_write(initial_action)
                actions_dict = self.setup_action_to_write_data(conv_action, p_idx)
                self.__write_data.update(actions_dict)
            self._write()
            self.__interface.mark_action_fulfilled(action_write_initial_data())

        t0 = time.time()
        self.__interface.initialize_data()
        print(f"RL-gym self.__interface.initialize_data() done in {time.time()-t0} seconds")
        # this results in reading data ahead of time when this is participat 2
        if self.__interface.is_read_data_available():
            self._read()

        self.__is_initialized = True
        self.__is_first_reset = False
        self.__t = 0
        self.__probes_rewards_data = {}
        self.__precice_read_data = {}

        # TODO: remove read_precice_observations call
        obs_list = self._read_precice_observations() + self.setup_observations()

        if return_info:
            return obs_list, {}

        return obs_list

    def step(self, action):
        if not self.__is_initialized:
            raise Exception("Call reset before interacting with the environment.")

        if self.__interface.is_action_required(
                action_write_iteration_checkpoint()):
            self.__interface.mark_action_fulfilled(
                action_write_iteration_checkpoint())

        self.__write_data = {}
        # dummy random values to be sent to the solver
        for p_idx in range(self.__options['n_parallel_env']):
            conv_action = self.setup_patch_field_to_write(action[p_idx])
            actions_dict = self.setup_action_to_write_data(conv_action, p_idx)
            self.__write_data.update(actions_dict)
        # print('inside step function --- just before advance===========================')
        # print(self.__write_data)

        self._advance()

        reward = self.setup_reward()
        done = not self.__interface.is_coupling_ongoing()

        # delete precice object upon done (a workaround to get precice reset)
        if done:
            self.__interface.finalize()
            del self.__interface
            print("preCICE finalized...\n")
            # we need to check here that solver run is finalized
            self.__solver_run = self._finalize_subprocess(self.__solver_run)

            # reset pointers
            self.__interface = None
            self.__solver_full_reset = False

        # TODO: remove read_precice_observations call as we most likely will not read anything from precice
        obs_list = self._read_precice_observations() + self.setup_observations()

        return obs_list, reward, done, {}

    def render(self, mode='human'):
        """ not implemented """
        if mode == "human":
            pass

    def close(self):
        """ not implemented """

    # preCICE related methods:
    def _init_precice(self):
        if self.__interface:
            raise Exception("precice interface already initialized, we should not reach here in normal situations")
        self.__interface = precice.Interface("RL-Gym", self.__options['precice_cfg'], 0, 1)

        self.__dim = self.__interface.get_dimensions()

        self.__mesh_id = {}
        self.__vertex_coords = {}
        self.__vertex_ids = {}
        for mesh_name in self.__mesh_list:
            mesh_id = self.__interface.get_mesh_id(mesh_name)
            print(f"Mesh_ID: {mesh_id}, Mesh_name: {mesh_name}")
            # bounding_box = [-np.inf, np.inf] * self.__dim
            # self.__interface.set_mesh_access_region(mesh_id, bounding_box)  # ERROR:  setMeshAccessRegion may only be called once.
            self.__mesh_id[mesh_name] = mesh_id

            vertex_coords = self.setup_mesh_coords(mesh_name)
            vertex_ids = self.__interface.set_mesh_vertices(mesh_id, vertex_coords)
            self.__vertex_ids[mesh_name] = vertex_ids
            self.__vertex_coords[mesh_name] = vertex_coords
        # establish connection with the solver
        self.__precice_dt = self.__interface.initialize()

    def _advance(self):
        self._write()
        self.__interface.advance(self.__precice_dt)
        # increase the time before reading the probes/forces for internal consistency checks
        self.__t += self.__precice_dt
        if not self.__options['is_dummy_run']:
            self._read_probes_rewards_files()
        # read precice after reading the files to avoid a nasty bug because of slow reading from files
        self._read()

    def _set_precice_data(self):
        self.__read_ids = {}
        self.__write_ids = {}

        # precice data from a single mesh on the solver side
        for mesh_name in self.__mesh_list:
            try:
                read_var_list = self.__mesh_variables[mesh_name]['read']
            except Exception as e:
                # a valid situation when the mesh doesn't have any read variables
                read_var_list = []
            try:
                write_var_list = self.__mesh_variables[mesh_name]['write']
            except Exception as e:
                # a valid situation when the mesh doesn't have any write variables
                write_var_list = []

            for read_var in read_var_list:
                self.__read_ids[read_var] = self.__interface.get_data_id(read_var, self.__mesh_id[mesh_name])

            for write_var in write_var_list:
                self.__write_ids[write_var] = self.__interface.get_data_id(write_var, self.__mesh_id[mesh_name])

            # print(mesh_name, self.__vertex_ids[mesh_name], self.__read_ids, self.__write_ids)

    def _read(self):
        if self.__interface.is_read_data_available():
            for mesh_name in self.__mesh_list:
                try:
                    read_var_list = self.__mesh_variables[mesh_name]['read']
                except Exception as e:
                    read_var_list = []
                for read_var in read_var_list:
                    if read_var in self.vector_variables:
                        self.__precice_read_data[read_var] = self.__interface.read_block_vector_data(
                            self.__read_ids[read_var], self.__vertex_ids[mesh_name])
                    else:
                        self.__precice_read_data[read_var] = self.__interface.read_block_scalar_data(
                            self.__read_ids[read_var], self.__vertex_ids[mesh_name])
                    print(f"(RL-Gym), avg-{read_var} using {mesh_name} read = {self.__precice_read_data[read_var].mean():.4f}")
                    print("-------------------------------------------")

    def _write(self):
        for mesh_name in self.__mesh_list:
            try:
                write_var_list = self.__mesh_variables[mesh_name]['write']
            except Exception as e:
                write_var_list = []
            for write_var in write_var_list:
                if write_var in self.vector_variables:
                    self.__interface.write_block_vector_data(
                        self.__write_ids[write_var], self.__vertex_ids[mesh_name], self.__write_data[write_var])
                else:
                    self.__interface.write_block_scalar_data(
                        self.__write_ids[write_var], self.__vertex_ids[mesh_name], self.__write_data[write_var])
                print(f"(RL-Gym), avg-{write_var} using {mesh_name} write = {self.__write_data[write_var].mean():.4f}")
                print("-------------------------------------------")

    def _launch_dummy_subprocess(self, process_idx, cwd):
        cmd_str = f'python -u fluid-solver.py'
        subproc = subprocess.Popen(cmd_str, shell=True, cwd=cwd)
        return subproc

    def _launch_subprocess(self, shell_cmd, cmd, cwd, cmd_type):
        cmd_str = f'. ../{shell_cmd} {cmd}'  # here we have relative path
        if cmd_type in ['clean', 'softclean']:
            try:
                completed_process = subprocess.run(cmd_str, shell=True, cwd=cwd)
            except Exception as e:
                print(e)
                raise Exception(f'failed to run {cmd_type}: {cmd_str} from the folder {cwd}')

            if completed_process.returncode != 0:
                raise Exception(f"run is not successful: {completed_process}")
            return None
        elif cmd_type == 'preprocess':
            # preprocess on the main folder before the symbolic links
            completed_process = subprocess.run(cmd_str, shell=True, cwd=cwd)
            if completed_process.returncode != 0:
                raise Exception(f"run is not successful: {completed_process}")
            return None
        else:
            subproc = subprocess.Popen(cmd_str, shell=True, cwd=cwd)
            return subproc

    def _check_subprocess(self, subproc, run_cmd, p_case_path):
        # check if the spawning process is successful
        if not psutil.pid_exists(subproc.pid):
            raise Exception(f'Error: subprocess failed to be launched: {run_cmd} run from {p_case_path}')

        # finalize the subprocess if it is terminated (normally/abnormally)
        if psutil.Process(subproc.pid).status() == psutil.STATUS_ZOMBIE:
            print(psutil.Process(subproc.pid), psutil.Process(subproc.pid).status())
            raise Exception(f'Error: subprocess failed to be launched: {run_cmd} STATUS_ZOMBIE run from {p_case_path}')

    def _finalize_subprocess(self, process_list):
        for subproc in process_list:
            if subproc and psutil.pid_exists(subproc.pid):
                if psutil.Process(subproc.pid).status() != psutil.STATUS_ZOMBIE:
                    print("subprocess status is not zombie - waiting to finish ...")
                    exit_signal = subproc.wait()
                else:
                    print("subprocess status is zombie - cleaning up ...")
                    exit_signal = subproc.poll()
                # check the subprocess exit signal
                if exit_signal != 0:
                    raise Exception("subprocess failed to complete its shell command: " + subproc.args)
                print("subprocess successfully completed its shell command: " + subproc.args)

                # # force to kill the subprocess if still around
                # self._kill_subprocess(subproc) #  not necessary, poll/wait should do the job!
        return []

    def _read_precice_observations(self):
        # TODO: delete this as we only read from post-processing files
        precice_obs_list = []
        for read_var in self.__precice_read_data.keys():
            precice_obs_list.append(self.__precice_read_data[read_var])
        return precice_obs_list

    def _get_observations_dict(self):
        if not self.__is_initialized:
            raise Exception("Call reset before interacting with the environment.")
        obs_dict = {}
        for field_ in self.__options['postprocessing_data'].keys():
            field_info = self.__options['postprocessing_data'][field_]
            if field_info['use'] == "observation" and \
                    field_ in self.__probes_rewards_data.keys() and \
                    len(self.__probes_rewards_data[field_]) > 0:
                # get the last n_parallel_env of the list as it orders by processor index
                obs_dict[field_] = self.__probes_rewards_data[field_][-self.__options['n_parallel_env']:]

        return obs_dict

    def _get_reward_dict(self):
        reward_dict = {}
        for field_ in self.__options['postprocessing_data'].keys():
            field_info = self.__options['postprocessing_data'][field_]
            if field_info['use'] == "reward" and \
                    field_ in self.__probes_rewards_data.keys() and \
                    len(self.__probes_rewards_data[field_]) > 0:
                # get the last n_env rows
                reward_dict[field_] = self.__probes_rewards_data[field_][-self.__options['n_parallel_env']:]

        return reward_dict

    def _read_probes_rewards_files(self):
        # sequential read of a single line (last line) of the file at each RL-Gym step
        for p_idx in range(self.__options['n_parallel_env']):
            p_case_path = self.__options['case_path'] + f'_{p_idx}'
            for field_ in self.__options['postprocessing_data'].keys():

                temp_filename = f"{p_case_path}{self.__options['postprocessing_data'][field_]['output_file']}"
                print(f'reading filename: {temp_filename}')
                if temp_filename not in self.__postprocessing_filehandler_dict.keys():
                    file_object = open(temp_filename, 'r')
                    self.__postprocessing_filehandler_dict[temp_filename] = file_object

                while True:
                    line_text = self.__postprocessing_filehandler_dict[temp_filename].readline()
                    if line_text == "":
                        break
                    assert len(line_text) > 0, 'read a single line but it is of length 0 !!'
                    line_text = line_text.strip()
                    is_comment, time_idx, n_probes, probe_data = parse_probe_lines(line_text)
                    if is_comment:
                        continue
                    print(f"time: {time_idx}, Number of probes {n_probes}, probes data {probe_data}")

                    if field_ not in self.__probes_rewards_data.keys():
                        self.__probes_rewards_data[field_] = []
                    self.__probes_rewards_data[field_].append([time_idx, n_probes, probe_data])
                    assert np.isclose(time_idx, self.__t), f"probes/forces data should be at the same time as RL-Gym: {time_idx} vs {self.__t}"
                    # only read one line and return
                    break

    def _close_postprocessing_files(self):
        for filename_ in self.__postprocessing_filehandler_dict.keys():
            file_object = self.__postprocessing_filehandler_dict[filename_]
            try:
                file_object.close()
            except Exception as e:
                print(f"error in closing probes/forces file: {e}")
                pass
        self.__postprocessing_filehandler_dict = {}

    # problem specific functions
    def setup_patch_field_to_write(self, action):
        """ Problem specific function """
        print(f"Prescribed action: FlowRate = {action[0]:.6f} [m/s3] on jet1")

        # convert volumetric flow rate to a sinusoidal profile on the interface
        avg_U = action[0] / np.sum(self.magSf)
        origin = np.array([0, 0, 0.005])
        d = np.array([1, 0, 0])
        w = 10
        r = np.zeros((self.Cf.shape[0], 3))
        theta = np.zeros(self.Cf.shape[0])
        U = np.zeros((self.Cf.shape[0], 3))

        # estimated flow rate based on the sinusoidal profile
        Q_calc = 0
        for i, c in enumerate(self.Cf):
            r[i] = (c - origin) / (np.sqrt((c - origin).dot((c - origin))))
            theta[i] = math.radians(90) - math.acos(np.dot(r[i], d))
            U[i] = avg_U * math.pi / 2 * math.cos(math.pi / math.radians(w) * theta[i]) * self.nf[i]
            Q_calc += U[i].dot(self.Sf[i])

        # correct velocity profile to enforce mass conservation
        Q_err = action[0] - Q_calc
        U_err = Q_err / np.sum(self.magSf) * self.nf
        U += U_err

        # return the velocity profile
        Q_final = 0
        for i, u in enumerate(U):
            Q_final += u.dot(self.Sf[i])

        if abs(Q_final - action[0]) < 1e-6:
            print(f"Set flow rate = {Q_final:.6f} [m/s3] on the interface")
            return U
        else:
            raise Exception('estimated velocity profile violates mass conservation')

    def setup_mesh_data(self, case_path):
        """ Problem specific function """
        """ parse polyMesh to get interface points:"""
        t0 = time.time()
        print('starting to parse FoamMesh')
        foam_mesh = FoamMesh(case_path)
        print(f'Done to parsing FoamMesh in {time.time()-t0} seconds')

        self.Cf = foam_mesh.boundary_face_centres(b'cylinder_jet1')        
        self.Sf, self.magSf, self.nf = foam_mesh.boundary_face_area(b'cylinder_jet1')
        # assert self.__n == self.Cf.shape[0], f'information obtained from the single mesh precice is not correct: size {self.__n}, {self.Cf.shape[0]}'
        # single_mesh = self.__vertex_coords[self.__mesh_list[0]]
        # if np.sum(np.abs(single_mesh - self.Cf)) > 1e-6:
        #     print(self.Cf)
        #     print(self.__vertex_coords[0])
        #     print(np.sum(np.abs(self.__vertex_coords[0] - self.Cf)))
        #     raise Exception('information obtained from the single mesh precice is not correct: grid')

    def setup_mesh_coords(self, mesh_name):
        """ Problem specific function """
        return self.Cf

    def setup_env_obs_act(self):
        # TODO: this should be problem specific
        # 1) action space need to be generalized here
        # 2) action space is not equal to observation space
        # 3) some of variables are scaler and other are vector
        # 4) where is rewards -- it is communicated? or read from online file?

        self.__n = 0
        for mesh_name in self.__mesh_list:
            self.__n += self.__vertex_coords[mesh_name].shape[0]
        # TODO: this should work even if RL-Gym have more than one mesh (not tested)
        self.__n = int(self.__n / self.__options['n_parallel_env'])
        print(f'size of the grid is {self.__n}')

        self.action_space = spaces.Box(
            low=0, high=0.0001, shape=(1, ), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.__n,), dtype=np.float64)

    def setup_action_to_write_data(self, action, p_idx=0):
        """ Problem specific function """
        """ return dummy random values for Velocity """
        write_data_dict = {f"Velocity_{p_idx}": action}
        return write_data_dict

    def setup_initial_action(self, p_idx):
        return [0, 0]

    def setup_observations(self):
        obs_dict = self._get_observations_dict()
        # now we print all of the observations to a list
        obs_list = []
        # we should be filtering here some of the columns only
        for field_ in obs_dict.keys():
            obs_list.append([field_, obs_dict[field_]])
        return obs_list

    def setup_reward(self):
        """ Problem specific function """
        reward_dict = self._get_reward_dict()
        # now we print all of the reward to a list
        rewards_list = []
        # we should be filtering here some of the columns only
        for field_ in reward_dict.keys():
            rewards_list.append([field_, reward_dict[field_]])
        return rewards_list
