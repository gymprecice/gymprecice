from cmath import inf
import subprocess
import argparse
import numpy as np

import precice
from precice import action_write_initial_data, \
    action_write_iteration_checkpoint

import gym
from gym import spaces


class FluidSolverAdapterEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self) -> None:
        super().__init__()

        self.__interface = None  # preCICE interface
        self.__solver_process = None
        self._init_precice()

        self.action_space = spaces.Box(
            low=5, high=10, shape=(self.__N + 1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=(self.__N + 1,), dtype=np.float64)

    # gym related methods:
    def step(self, action):
        s = self.state
        assert s is not None,\
            "Call reset before interacting with the environment."

        if self.__interface.is_action_required(
                action_write_iteration_checkpoint()):
            self.__interface.mark_action_fulfilled(
                action_write_iteration_checkpoint())

        # dummy randon values to be sent to the solver
        self.__CrossSectionLength = self._calc_cross_section_length(action)

        self._advance()

        reward = -(self.__pressure.mean() - self.__opt_pressure)**2
        done = not self.__interface.is_coupling_ongoing()

        # delete precice object upon done (a workaround to get precice reset)
        if done:
            self.__interface.finalize()
            del self.__interface
            self.__interface = None
            print("preCICE finalised...\n")

        return self._get_obs(), reward, done, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # initiate precice interface
        self._init_precice()
        
        # launch the solver
        cmd_solver = options["cmd_solver"]

        if self.__solver_process is None:
            self.__solver_process = subprocess.Popen(cmd_solver, shell=True)
        else:
            self.__solver_process.wait()  # this should terminate the process
            self.__solver_process = subprocess.Popen(cmd_solver, shell=True)

        # establish connection with the solver
        self.__precice_dt = self.__interface.initialize()

        if self.__interface.is_action_required(action_write_initial_data()):
            self._write()
            self.__interface.mark_action_fulfilled(action_write_initial_data())

        self.__interface.initialize_data()

        if self.__interface.is_read_data_available():
            self._read()

        print("-------------------------------------------")
        print("Received: initial (mean)-pressure from Solver= %.2f"
              % self.__pressure.mean())

        self.state = 1.0
        return self._get_obs()

    def close(self):
        pass

    def _get_obs(self):
        s = self.state
        assert s is not None, \
            "Call reset before interacting with the environment."
        return self.__pressure

    # preCICE related methods:

    def _init_precice(self):
        if self.__interface is None:
            self._config_precice()
            self.__interface = precice.Interface(
                "Adapter", self.__args.configurationFileName, 0, 1)
            self._set_data_precice()

    def _config_precice(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "configurationFileName", help="Name of the xml config file.",
            nargs='?', type=str, default="precice-config.xml")

        try:
            self.__args = parser.parse_args()
        except SystemExit:
            print("")
            print("Did you forget adding the precice configuration file as an  \
                    argument?")
            quit()
        print("\npreCICE configured...")

    def _set_data_precice(self):
        self.__t = 0.0
        L = 10  # length of tube/simulation domain
        self.__N = 100  # number of elements in x direction
        self.__opt_pressure = 100  # dummuy optimum average pressure in tube

        self.__dimensions = self.__interface.get_dimensions()
        self.__grid = np.zeros([self.__N + 1, self.__dimensions])
        self.__grid[:, 0] = np.linspace(0, L, self.__N + 1)  # x component
        self.__grid[:, 1] = 0  # y component
        self.__vertexIDs = np.zeros(self.__N + 1)
        self.__meshID = self.__interface.get_mesh_id("Adapter-Mesh")
        self.__vertexIDs = self.__interface.set_mesh_vertices(
            self.__meshID, self.__grid)

        self.__pressureID = self.__interface.get_data_id(
            "Pressure", self.__meshID)
        self.__pressure = np.ones(self.__N + 1)
        self.__CrossSectionLengthID = self.__interface.get_data_id(
                "CrossSectionLength", self.__meshID)
        self.__CrossSectionLength = np.random.uniform(0, 10, self.__N + 1)

    def _read(self):
        if self.__interface.is_read_data_available():
            self.__pressure = self.__interface.read_block_scalar_data(
                self.__pressureID, self.__vertexIDs)

    def _write(self):
        self.__interface.write_block_scalar_data(
            self.__CrossSectionLengthID, self.__vertexIDs,
            self.__CrossSectionLength)

    def _advance(self):
        print("-------------------------------------------")
        print("Sent: (mean)-cross sectional length to solver = %.2f"
              % self.__CrossSectionLength.mean())
        self._write()
        self.__interface.advance(self.__precice_dt)
        self._read()
        print("Received: (mean)-pressure from Solver = %.2f"
              % self.__pressure.mean())
        self.__t += self.__precice_dt

    def _calc_cross_section_length(self, action):
        # return dummy random values
        return np.random.uniform(5, 10, self.__N + 1)
