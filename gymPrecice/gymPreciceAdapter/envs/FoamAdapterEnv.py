from cmath import inf
import subprocess
import argparse
import numpy as np

import precice
from precice import action_write_initial_data, \
    action_write_iteration_checkpoint

import gym
from gym import spaces
from .mesh_parser import FoamMesh


class FoamAdapterEnv(gym.Env):
    """
    precice adapter class to couple gym and open-foam
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self) -> None:
        super().__init__()

        # gym env attributes:
        self.state = None
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64
        )

        # coupling attributes:
        self.__precice_dt = None
        # self.__t = None
        self.__args = None
        self.__grid = None
        self.__mesh_id = None
        self.__vertex_id = None
        self.__temperature = None
        self.__temperature_id = None
        self.__heat_flux = None
        self.__heat_flux_id = None
        # self.__interfaces = np.array(["interface", "top"])
        self.__n = None

        self.__interface = None  # preCICE interface
        self.__solver_process = None  # physical solver
        self.__solver_full_reset = False
        self._init_precice()

    # gym related methods:
    def step(self, action):
        state = self.state
        assert state is not None,\
            "Call reset before interacting with the environment."
        if self.__interface.is_action_required(
                action_write_iteration_checkpoint()):
            self.__interface.mark_action_fulfilled(
                action_write_iteration_checkpoint())

        # dummy random values to be sent to the solver
        self.__heat_flux = self._calc_heat_flux(action)

        self._advance()

        reward = 1.0
        done = not self.__interface.is_coupling_ongoing()

        # delete precice object upon done (a workaround to get precice reset)
        if done:
            self.__interface.finalize()
            del self.__interface
            self.__interface = None
            if self.__solver_full_reset:
                self.__solver_process.wait()
                self.__solver_process = None
            print("preCICE finalised...\n")

        return self._get_obs(), reward, done, {}

    def reset(self, *, seed=None, return_info=False, options):
        super().reset(seed=seed)

        self._init_precice()

        # get the solver-launch options
        src_cmd = ". " + options.get("src_cmd", "")
        case_path = options.get("case_path", "")
        preprocessor_cmd = options.get("preprocess_cmd", "")
        run_cmd = options.get("run_cmd", "")
        preprocess_log = options.get("preprocess_log", "")
        last_run_log = options.get("run_log", "")
        self.__solver_full_reset = options.get(
            "solver_full_reset", self.__solver_full_reset)

        if self.__solver_process is None:
            # separate preprocessor and run to make sure the data is
            # available for foam-parser
            preprocessor_cmd = src_cmd + " && " + preprocessor_cmd + " " \
                + case_path + " > " + case_path + "/" + preprocess_log \
                + " 2>&1"

            run_cmd = src_cmd + " && " + run_cmd + " " + case_path \
                + " > " + case_path + "/" + last_run_log + " 2>&1"

            # run open-foam preprocessor
            preprocess = subprocess.Popen(preprocessor_cmd, shell=True)
            preprocess.wait()  # wait till pre-process is done

            # parse solver mesh data
            self._parse_mesh_data(case_path)
            # set mesh-dependent gym data
            self._set_env_obs_act()

            # run open-foam solver
            self.__solver_process = subprocess.Popen(run_cmd, shell=True)
        else:
            run_cmd = src_cmd + " && " + run_cmd + " " + case_path \
                + " > " + case_path + "/" + last_run_log + " 2>&1"
            if not self.__solver_full_reset:
                self.__solver_process.wait()  # wait to terminate

            # run open-foam solver
            self.__solver_process = subprocess.Popen(run_cmd, shell=True)

        self._set_precice_data()

        # establish connection with the solver
        self.__precice_dt = self.__interface.initialize()

        if self.__interface.is_action_required(action_write_initial_data()):
            self._write()
            self.__interface.mark_action_fulfilled(action_write_initial_data())

        self.__interface.initialize_data()

        if self.__interface.is_read_data_available():
            self._read()
        print("-------------------------------------------")
        print(f"avg-Temperature from Solver = {self.__temperature.mean():.2f}")

        self.state = 1.0
        return self._get_obs()

    def render(self, mode='human'):
        """ not implemented """
        if mode == "human":
            pass

    def close(self):
        """ not implemented """
        pass

    def _set_env_obs_act(self):
        """
        set mesh-dependentgym env data:
        """
        self.action_space = spaces.Box(
            low=-inf, high=inf, shape=(self.__n,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=(self.__n,), dtype=np.float64)

    def _get_obs(self):
        state = self.state
        assert state is not None, \
            "Call reset before interacting with the environment."
        return self.__temperature

    # preCICE related methods:
    def _init_precice(self):
        if self.__interface is None:
            self._config_precice()
            self.__interface = precice.Interface(
                "Adapter", self.__args.configurationFileName, 0, 1)

    def _config_precice(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "configurationFileName", help="Name of the xml config file.",
            nargs='?', type=str, default="precice-config.xml")
        try:
            self.__args = parser.parse_args()
        except SystemExit:
            print("")
            print("Did you forget adding the precice configuration file as an \
                    argument?")
            quit()
        print("\npreCICE configured...")

    def _advance(self):
        print("-------------------------------------------")
        print(f"avg-HeatFlux to solver = {self.__heat_flux.mean():.2f}")
        self._write()

        self.__interface.advance(self.__precice_dt)

        self._read()
        print(f"avg-Temperature from Solver = {self.__temperature.mean():.2f}")
        # self.__t += self.__precice_dt

    def _set_precice_data(self):
        # precice data:
        # self.__t = 0.0
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
        vertices = foam_mesh.boundary_vertices(b'interface')
        self.__n = vertices.shape[0]
        self.__grid = vertices

    def _read(self):
        if self.__interface.is_read_data_available():
            self.__temperature = self.__interface.read_block_scalar_data(
                self.__temperature_id, self.__vertex_id)

    def _write(self):
        self.__interface.write_block_scalar_data(
            self.__heat_flux_id, self.__vertex_id,
            self.__heat_flux)

    def _calc_heat_flux(self, action):
        """ return dummy random values for heat_flux """
        return np.random.uniform(100, 10000, self.__n)
