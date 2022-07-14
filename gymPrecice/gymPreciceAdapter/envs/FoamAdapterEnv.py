from cmath import inf
from signal import signal
import subprocess
import argparse
from tkinter.messagebox import NO
import numpy as np

import precice
from precice import action_write_initial_data, \
    action_write_iteration_checkpoint

import gym
from gym import spaces
from .mesh_parser import FoamMesh

import time
import psutil


class FoamAdapterEnv(gym.Env):
    """
    precice adapter class to couple gym and open-foam
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self) -> None:
        super().__init__()

        # gym env attributes:
        self.__is_initalized = False
        self.__is_first_reset = True  # if True, gym env reset has been called at least once
        # action_ and observation_space will be set later in _set_env_obs_act()
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64
        )

        # coupling attributes:
        self.__precice_dt = None
        # self.__t = None
        self.__interface = None  # preCICE interface
        self.__args = None
        self.__grid = None
        self.__mesh_id = None
        self.__dim = None
        self.__vertex_id = None
        self.__temperature = None
        self.__temperature_id = None
        self.__heat_flux = None
        self.__heat_flux_id = None
        # self.__interfaces = np.array(["interface", "top"])
        self.__n = None

        # solver attributes:
        self.__solver_preprocess = None
        self.__solver_run = None  # physical solver
        self.__solver_full_reset = False  # if True, run foam-preprocessor upon every reset

    # gym methods:

    def reset(self, *, seed=None, return_info=True, options=None):
        super().reset(seed=seed)

        # initiate precice interface
        self._init_precice()

        # get the solver-launch options
        case_path = options.get("case_path", "")
        preprocessor_cmd = options.get("preprocess_cmd", "")
        run_cmd = options.get("run_cmd", "")
        self.__solver_full_reset = options.get(
            "solver_full_reset", self.__solver_full_reset)

        if (self.__is_first_reset or self.__solver_full_reset):
            # run open-foam preprocessor
            self.__solver_preprocess = self._launch_subprocess(preprocessor_cmd)
            self.__solver_preprocess = self._finalize_subprocess(self.__solver_preprocess)
            assert self.__solver_preprocess is None

            # parse solver mesh data
            self._parse_mesh_data(case_path)
            # set mesh-dependent gym data (observation & action)
            self._set_env_obs_act()

            # run open-foam solver
            if self.__solver_run:
                raise Exception('solver_run pointer is not cleared -- should not reach here')
            self.__solver_run = self._launch_subprocess(run_cmd)
            assert self.__solver_run is not None
        else:
            # run open-foam solver
            if self.__solver_run:
                raise Exception('solver_run pointer is not cleared -- should not reach here')
            self.__solver_run = self._launch_subprocess(run_cmd)
            assert self.__solver_run is not None

        self._set_precice_data()

        # establish connection with the solver
        self.__precice_dt = self.__interface.initialize()

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
            return self._get_obs(), {}
        return self._get_obs()

    def step(self, action):
        if not self.__is_initalized:
            raise Exception("Call reset before interacting with the environment.")

        if self.__interface.is_action_required(
                action_write_iteration_checkpoint()):
            self.__interface.mark_action_fulfilled(
                action_write_iteration_checkpoint())

        # dummy random values to be sent to the solver
        self.__heat_flux = self._calc_heat_flux(action)

        self._advance()

        reward = self._calc_reward()
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

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        """ not implemented """
        if mode == "human":
            pass

    def close(self):
        """ not implemented """

    def _set_env_obs_act(self):
        """
        set mesh-dependentgym env data:
        """
        self.action_space = spaces.Box(
            low=100000, high=200000, shape=(self.__n,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=(self.__n,), dtype=np.float64)

    def _calc_reward(self):
        # In a simulation enviroment there are two type of observations:
        # 1- Observations for control (real observations)
        # 2- Observation/states for estimating rewards (e.g. drag or lift forces)
        return 1.0

    def _get_obs(self):
        if not self.__is_initalized:
            raise Exception("Call reset before interacting with the environment.")

        #self.observation_space = self.__temperature
        return self.__temperature

    # preCICE related methods:
    def _init_precice(self):
        if self.__interface:
            raise Exception("precice interface already initalized, we should not reach here in normal situations")
        self._config_precice()
        self.__interface = precice.Interface(
            "Adapter", self.__args.configurationFileName, 0, 1)

    def _config_precice(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "configurationFileName", help="Name of the xml config file.",
            nargs='?', type=str, default="../precice-config.xml")

        try:
            self.__args = parser.parse_args()
        except:
            raise Exception ("Add the precice configuration file as argument")
        print("\npreCICE configured...")

    def _advance(self):
        self._write()
        self.__interface.advance(self.__precice_dt)
        self._read()
        # self.__t += self.__precice_dt

    def _set_precice_data(self):
        # precice data:
        # self.__t = 0.0
        self.__dim = self.__interface.get_dimensions()
        self.__mesh_id = self.__interface.get_mesh_id("Adapter-Mesh")
        self.__vertex_id = self.__interface.set_mesh_vertices(
            self.__mesh_id, self.__grid)

        self.__temperature_id = self.__interface.get_data_id(
            "Temperature", self.__mesh_id)
        self.__temperature = np.zeros(self.__n)
        self.__heat_flux_id = self.__interface.get_data_id(
                "Heat-Flux", self.__mesh_id)
        self.__heat_flux = np.zeros(self.__n)

    def _parse_mesh_data(self, case_path):
        """ parse polyMesh to get interface points:"""
        foam_mesh = FoamMesh(case_path)

        # nodes = foam_mesh.boundary_face_nodes(b'interface')
        # self.__n = nodes.shape[0]
        # self.__grid = nodes

        faces = foam_mesh.boundary_face_centres(b'interface')
        self.__n = faces.shape[0]
        self.__grid = faces

    def _read(self):
        if self.__interface.is_read_data_available():
            self.__temperature = self.__interface.read_block_scalar_data(
                self.__temperature_id, self.__vertex_id)
            print(f"avg-Temperature from Solver = {self.__temperature.mean():.2f}")
            self._print_patch_data(self.__temperature, self.__grid)

    def _write(self):
        self.__interface.write_block_scalar_data(
            self.__heat_flux_id, self.__vertex_id,
            self.__heat_flux)
        print(f"avg-HeatFlux to solver = {self.__heat_flux.mean():.2f}")
        print("-------------------------------------------")

    def _calc_heat_flux(self, action):
        """ return dummy random values for heat_flux """
        return action

    def _print_patch_data(self, data, patch):
        idx = 1
        for value in data:
            p = patch[idx-1]
            if idx % 5 != 0:
                print(f"[{idx-1}, ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {value:.4f}]", end=" ")
            else:
                print(f"[{idx-1}, ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {value:.4f}]", end="\n")
            idx += 1

    def _launch_subprocess(self, cmd):
        subproc = subprocess.Popen(cmd, shell=True)

        # check if the command is working
        time.sleep(0.05)  # have to add a very tiny sleep

        # check if the spawning process is sucessful
        if not psutil.pid_exists(subproc.pid):
            raise Exception('Error: subprocess failed to be launched: ' + cmd)
        
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
