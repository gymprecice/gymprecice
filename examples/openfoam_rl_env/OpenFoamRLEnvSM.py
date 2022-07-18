# from cmath import inf
from signal import signal
import subprocess
import argparse
from tkinter.messagebox import NO
import numpy as np

import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint

import gym
from gym import spaces
from mesh_parser import FoamMesh

import time
import psutil
from utils import get_cfg_data, parse_probe_lines


class OpenFoamRLEnv(gym.Env):
    """
    RL Env via a dummy precice adapter class
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, options) -> None:
        super().__init__()

        self.__options = options

        # gym env attributes:
        self.__is_initalized = False
        self.__is_first_reset = True  # if True, gym env reset has been called at least once
        # action_ and observation_space will be set in _set_precice_data
        self.action_space = None
        self.observation_space = None

        scaler_variables, vector_variables, mesh_list, mesh_variables = \
            get_cfg_data('', self.__options['precice_cfg'])
        self.__mesh_variables = mesh_variables
        self.__mesh_list = mesh_list
        # scaler and vector variables should be used to define the size of action space
        self.__scaler_variables = scaler_variables
        self.vector_variables = vector_variables

        # coupling attributes:
        self.__precice_dt = None
        # self.__t = None
        self.__interface = None  # preCICE interface
        self.__mesh_id = None
        self.__dim = None
        self.__vertex_id = None
        # # self.__interfaces = np.array(["interface", "top"])
        # self.__n = None

        # solver attributes:
        self.__solver_preprocess = None
        self.__solver_run = None  # physical solver
        self.__solver_full_reset = False  # if True, run foam-preprocessor upon every reset
        self.__filehandler_dict = {}
        # this should be parsed from the OpenFOAM case files
        self.__probes_filename_list = [
            f"{options['case_path']}/postProcessing/probes/0/U",
            f"{options['case_path']}/postProcessing/probes/0/T",
            f"{options['case_path']}/postProcessing/probes/0/p",
        ]

    # gym methods:
    def reset(self, *, seed=None, return_info=True, options=None):
        super().reset(seed=seed)

        # get the solver-launch options
        case_path = options.get("case_path", "")
        preprocessor_cmd = options.get("preprocess_cmd", "")
        run_cmd = options.get("run_cmd", "")
        self.__solver_full_reset = options.get(
            "solver_full_reset", self.__solver_full_reset)

        if (self.__is_first_reset or self.__solver_full_reset):
            # run open-foam preprocessor
            self.__solver_preprocess = self._launch_subprocess(preprocessor_cmd, case_path)
            self.__solver_preprocess = self._finalize_subprocess(self.__solver_preprocess)
            assert self.__solver_preprocess is None

            # # parse solver mesh data
            # self._parse_mesh_data(case_path)
            # # set mesh-dependent gym data (observation & action)
            # self._set_env_obs_act()

        if len(self.__filehandler_dict) > 0:
            self.close_probes_rewards_files()
            self.__filehandler_dict = {}

        # run open-foam solver
        if self.__solver_run:
            raise Exception('solver_run pointer is not cleared -- should not reach here')
        self.__solver_run = self._launch_subprocess(run_cmd, case_path)
        assert self.__solver_run is not None

        # initiate precice interface and read single mesh data
        self._init_precice()
        self._set_precice_data()
        self.define_env_obs_act()
        # debuging for the now
        self._parse_mesh_data(case_path)

        if self.__interface.is_action_required(action_write_initial_data()):
            self._write()
            self.__interface.mark_action_fulfilled(action_write_initial_data())

        self.__interface.initialize_data()

        if self.__interface.is_read_data_available():
            print("-------------------------------------------")
            self._read()

        self.__is_initalized = True
        self.__is_first_reset = False

        if return_info:
            return self.read_data_to_observations(), {}
        return self.read_data_to_observations()

    def step(self, action):
        if not self.__is_initalized:
            raise Exception("Call reset before interacting with the environment.")

        if self.__interface.is_action_required(
                action_write_iteration_checkpoint()):
            self.__interface.mark_action_fulfilled(
                action_write_iteration_checkpoint())

        # dummy random values to be sent to the solver
        self.__write_data = self.action_to_write_data(action)

        self._advance()

        reward = self.get_reward()
        done = not self.__interface.is_coupling_ongoing()

        # delete precice object upon done (a workaround to get precice reset)
        if done:
            self.__interface.finalize()
            del self.__interface
            print("preCICE finalised...\n")
            # we need to check here that solver run is finalized
            self.__solver_run = self._finalize_subprocess(self.__solver_run)

            # reset pointers
            self.__interface = None
            self.__solver_full_reset = False

        return self.read_data_to_observations(), reward, done, {}

    def render(self, mode='human'):
        """ not implemented """
        if mode == "human":
            pass

    def close(self):
        """ not implemented """

    # preCICE related methods:
    def _init_precice(self):
        if self.__interface:
            raise Exception("precice interface already initalized, we should not reach here in normal situations")
        self.__interface = precice.Interface("RL-Gym", self.__options['precice_cfg'], 0, 1)

        self.__dim = self.__interface.get_dimensions()

        self.__mesh_id = {}
        for mesh_name in self.__mesh_list:
            mesh_id = self.__interface.get_mesh_id(mesh_name)
            bounding_box = [-np.inf, np.inf] * self.__dim
            self.__interface.set_mesh_access_region(mesh_id, bounding_box)
            self.__mesh_id[mesh_name] = mesh_id

        # establish connection with the solver
        self.__precice_dt = self.__interface.initialize()

    def _advance(self):
        self._write()
        self.__interface.advance(self.__precice_dt)
        self._read()
        self.read_probes_rewards_files()

        # self.__t += self.__precice_dt

    def _set_precice_data(self):
        self.__vertex_coords = {}
        self.__vertex_id = {}
        self.__read_id = {}
        self.__write_id = {}

        # precice data from a single mesh on the solver side
        for mesh_name in self.__mesh_list:
            vertex_id, vertex_coords = self.__interface.get_mesh_vertices_and_ids(self.__mesh_id[mesh_name])
            self.__vertex_id[mesh_name] = vertex_id
            self.__vertex_coords[mesh_name] = vertex_coords
            # print(self.__mesh_variables)
            # print(self.__mesh_variables[mesh_name])
            read_var_list = self.__mesh_variables[mesh_name]['read']
            write_var_list = self.__mesh_variables[mesh_name]['write']
            for read_var in read_var_list:
                self.__read_id[read_var] = self.__interface.get_data_id(read_var, self.__mesh_id[mesh_name])

            for write_var in write_var_list:
                self.__write_id[write_var] = self.__interface.get_data_id(write_var, self.__mesh_id[mesh_name])

            print(mesh_name, self.__vertex_id[mesh_name], self.__read_id[read_var], self.__write_id[write_var])
            print("grid obtained from precice single mesh: ")
            print(vertex_coords)
            print(vertex_coords.shape, vertex_id.shape)
            print("---- recived all the data needed from the fluid solver")

    def _read(self):
        self.__read_data = {}
        if self.__interface.is_read_data_available():
            for mesh_name in self.__mesh_list:
                read_var_list = self.__mesh_variables[mesh_name]['read']
                for read_var in read_var_list:
                    self.__read_data[read_var] = self.__interface.read_block_scalar_data(
                        self.__read_id[read_var], self.__vertex_id[mesh_name])
                    print(f"avg-{read_var} from Solver = {self.__read_data[read_var].mean():.4f}")
                    print("-------------------------------------------")

    def _write(self):
        for mesh_name in self.__mesh_list:
            write_var_list = self.__mesh_variables[mesh_name]['write']
            for write_var in write_var_list:
                self.__interface.write_block_scalar_data(
                    self.__write_id[write_var], self.__vertex_id[mesh_name],
                    self.__write_data[write_var])
                print(f"avg-{write_var} to solver = {self.__write_data[write_var].mean():.4f}")
                print("-------------------------------------------")

    def _launch_subprocess(self, cmd, cwd=None):
        subproc = subprocess.Popen(cmd, shell=True, cwd=cwd)

        # check if the command is working
        time.sleep(0.5)  # have to add a very tiny sleep

        # check if the spawning process is sucessful
        if not psutil.pid_exists(subproc.pid):
            raise Exception(f'Error: subprocess failed to be launched: {cmd}')

        # finalize the subprocess if it is terminated (normally/abnormally)
        if psutil.Process(subproc.pid).status() == psutil.STATUS_ZOMBIE:
            return self._finalize_subprocess(subproc)
        return subproc

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
                raise Exception("subprocess failed to complete its shell command: " + subproc.args)
            print("subprocess successfully completed its shell command: " + subproc.args)

            # # force to kill the subprocess if still around
            # self._kill_subprocess(subproc) #  not necessary, poll/wait should do the job!
        return None

    def _parse_mesh_data(self, case_path):
        """ parse polyMesh to get interface points:"""
        foam_mesh = FoamMesh(case_path)

        # nodes = foam_mesh.boundary_face_nodes(b'interface')
        # self.__n = nodes.shape[0]
        # self.__grid = nodes

        if len(self.__mesh_list) > 1:
            raise Exception('currently this only works for single mesh used for both observations and actions')

        faces = foam_mesh.boundary_face_centres(b'interface')
        assert self.__n == faces.shape[0], 'information obtained from the single mesh precice is not correct: size'
        single_mesh = self.__vertex_coords[self.__mesh_list[0]]
        if np.sum(np.abs(single_mesh - faces)) > 1e-6:
            print(faces)
            print(self.__vertex_coords[0])
            print(np.sum(np.abs(self.__vertex_coords[0] - faces)))
            raise Exception('information obtained from the single mesh precice is not correct: grid')

    def define_env_obs_act(self):
        # TODO: this should be problem specific
        # 1) action space need to be generalized here 
        # 2) action space is not equal to observation space
        # 3) some of variables are scaler and other are vector
        # 4) where is rewards -- it is communicated? or read from online file?

        self.__n = 0
        for mesh_name in self.__mesh_list:
            self.__n += self.__vertex_coords[mesh_name].shape[0]
        print(f'size of the grid is {self.__n}')

        self.action_space = spaces.Box(
            low=100000, high=200000, shape=(self.__n,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.__n,), dtype=np.float64)

    def get_reward(self):
        # In a simulation enviroment there are two type of observations:
        # 1- Observations for control (real observations)
        # 2- Observation/states for estimating rewards (e.g. drag or lift forces)
        return 1.0

    def action_to_write_data(self, action):
        """ return dummy random values for heat_flux """
        write_data_dict = {"Heat-Flux": action}
        return write_data_dict

    def read_data_to_observations(self):
        if not self.__is_initalized:
            raise Exception("Call reset before interacting with the environment.")

        # self.observation_space = self.__temperature
        return self.__read_data["Temperature"]

    def read_probes_rewards_files(self):
        # file names and how the data should be used could be obtained by parsing the probes dict
        # read line by line at each loop
        for temp_filename in self.__probes_filename_list:
            print(f'reading filename: {temp_filename}')
            if temp_filename not in self.__filehandler_dict.keys():
                file_object = open(temp_filename, 'r')
                self.__filehandler_dict[temp_filename] = file_object

            while True:
                line_text = self.__filehandler_dict[temp_filename].readline()
                if line_text == "":
                    break
                line_text = line_text.strip()
                if len(line_text) > 0:
                    is_comment, time_idx, n_probes, probe_data = parse_probe_lines(line_text)
                if not is_comment:
                    print(f"time: {time_idx}, Number of probes {n_probes}, probes data {probe_data}")

        # read all the lines from the start at each step
        # temp_filename = '/data/ahmed/rl_play/examples/openfoam_rl_env/fluid-openfoam/postProcessing/probes/0/U'
        # with open(temp_filename, 'r') as filehandle:
        #     foam_text = filehandle.readlines()
        #     foam_text = [x.strip() for x in foam_text]
        #     foam_text = '\n'.join(foam_text)
        #     print(foam_text)

    def close_probes_rewards_files(self):
        for filename_ in self.__filehandler_dict.keys():
            file_object = self.__filehandler_dict[filename_]
            try:
                file_object.close()
            except Exception as e:
                print(e)
                pass

    def __del__(self):
        # close all the open files
        self.close_probes_rewards_files()
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



    # def _print_patch_data(self, data, patch):
    #     idx = 1
    #     for value in data:
    #         p = patch[idx - 1]
    #         if idx % 5 != 0:
    #             print(f"[{idx-1}, ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {value:.4f}]", end=" ")
    #         else:
    #             print(f"[{idx-1}, ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {value:.4f}]", end="\n")
    #         idx += 1

    # def _config_precice(self):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument(
    #         "configurationFileName", help="Name of the xml config file.",
    #         nargs='?', type=str, default="precice-config.xml")  # default="../precice-config.xml")

    #     try:
    #         self.__args = parser.parse_args()
    #     except Exception as e:
    #         print(e)
    #         raise Exception("Add the precice configuration file as argument")
    #     print("\npreCICE configured...")
    # def _kill_subprocess(self, subproc):
    #         if psutil.pid_exists(subproc.pid):
    #             try:
    #                 self._psutil_kill(subproc.pid)  # clean the <defunct> shell process
    #             except Exception as exception:
    #                 raise Exception(f'failed to kill defunct process: {exception}')

    # # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    # # kill both the shell and child process (solver command)
    # def _psutil_kill(self, proc_pid):
    #     process = psutil.Process(proc_pid)
    #     for proc in process.children(recursive=True):
    #         proc.kill()
    #     process.kill()
