"""Core API for environments."""
import logging
import math
import os
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TypeVar

import gymnasium as gym
import numpy as np
import precice
import psutil
from precice import (
    action_read_iteration_checkpoint,
    action_write_initial_data,
    action_write_iteration_checkpoint,
)

from gymprecice.utils.fileutils import make_env_dir
from gymprecice.utils.xmlutils import get_episode_end_time, get_mesh_data


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Adapter(ABC, gym.Env):
    r"""The main Gym-preCICE class for coupling Reinforcement Learning (RL) controllers and PDE-based numerical solvers.

    Gym-preCICE adapter is a generic base class that provides a common Gymnasium (aka "OpenAI Gym")-like API to couple single- or multi-physics
    numerical solvers (referred to as "physics simulation engine") and Reinforcement Learning Agents (referred to as "controller") using
    preCICE coupling library.

    The main application of Gym-preCICE adapter lies in closed- and open-loop active control of physics simulations.

    To control any case-specific physics simulation engine supported by preCICE, users of Gym-preCICE adapter need to define a
    class (referred to as "environment") that inherits from this adapter and override its four abstract methods to adapt with the underlying behaviour
    of the "behind-the-scene" physics simulation engine.

    The main abstract API methods that users of this adapter need to know and override in a case-specific environment are:

    - :meth:`_get_action` - Maps actions received from the controller into appropriate values or boundary fields to be communicated
      with the physics simulation engine.
      Returns a dictionary containing all bouundry values need to be communicated with the physics simulation engine via preCICE.
    - :meth:`_get_observation` - Maps data read from the physics simulation engine to appropriate observation input for the controller.
      Returns an element of the environment's :attr:`observation_space`, e.g. a numpy array containing probe (sensor) pressure data within a
      controlled flow field.
    - :meth:`_get_reward` - Computes and returns an instantaneous reward signal (a scalar value) achieved as a result of taking an action in
      the physics simulation engine.
    - :meth:`_close_external_resources` - Closes resources used by the physics simulation engine, e.g. a probe file or a database.

    Gym-preCICE adapter overrides three of the main Gymnasium API methods to allow communication between the controller and the environment:

    - :meth:`reset` - Establishes a connection between the controller and the environment via preCICE, and resets the environment to an initial state.
      Returns the first observation for the controller in an episode.
    - :meth:`step` - Updates the environment state using actions received from the controller.
      Returns the next observation for the controller, an instantaneous reward signal, and if the environment has terminated due to the latest action.
    - :meth:`close` - Closes the environment by switching off the coupling and releasing all resources used by preCICE and the physics simulation engine.

    """

    def __init__(self, options: dict = None, idx: int = 0) -> None:
        """Setup generic attributes.

        Args:
            options (dict): environment configuration.
            idx (int): environment index number.
        """
        try:
            self._precice_cfg = options["precice"]["config_file"]
            self._solver_list = options["solvers"]["name"]
            self._reset_script = options["solvers"]["reset_script"]
            self._prerun_script = options["solvers"].get(
                "prerun_script", self._reset_script
            )
            self._run_script = options["solvers"]["run_script"]
            self._actuator_list = options["actuators"]["name"]
        except KeyError as err:
            logger.error(f"Invalid key {err} in options")
            raise err

        self.action_space = None
        self.observation_space = None
        self._idx = idx
        self._env_dir = f"env_{self._idx}"
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
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        r"""Resets the environment, couples the environment with the controller, and returns the initial observation.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
            Please refer to https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py.
            options (optional dict): Additional information to specify how the environment is reset.

        Returns:
            observation (ObsType): Observation of the initial state.
            info (dictionary): his dictionary contains auxiliary information complementing ``observation``.
            It should be analogous to the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed, options=options)

        logger.debug(f"Reset {self._env_dir} - start")

        if self._first_reset is True:
            self._launch_subprocess("prerun_solvers")
            self._first_reset = False

        self._close_external_resources()
        # (1) reset the physics simulation engine
        self._launch_subprocess("reset_solvers")
        assert self._solver is None, "solver_run pointer is not cleared!"
        # (2) start the physics simulation engine as a subprocess
        p_process = self._launch_subprocess("run_solvers")
        assert p_process is not None, "slover launch failed!"
        self._solver = p_process
        self._check_subprocess_exists(self._solver)
        # (3) couple the physics simulation engine and the controller via preCICE, and run the physics simulation engine one-step forward
        self._init_precice()
        # (4) receive partial observation from the physics simulation engine
        obs = self._get_observation()

        logger.debug(f"Reset {self._env_dir} - end")

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        r"""Runs one timestep of the environment dynamics using the controller actions.

        Args:
            action (ActType): an action provided by the controller to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the controller actions.
                An example is a numpy array containing spatial pressure and velocity data collected from a simulated fluid flow domain.
            reward (float): The reward as a result of taking the action.
            terminated (bool): Whether the physics simulation engine reaches the end of simulation time. If true, :meth:`reset` needs to be called.
            truncated (bool): N/A for our environments.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                For more info on Args and Returns please refer to https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py.
        """
        logger.debug(f"Step {self._env_dir} - start")

        assert self._is_reset, "Call reset before using step method."
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        # (1) map the controller actions to boundary fields for the physics simulation engine
        write_data = self._get_action(action, self._write_var_list)
        # (2) complete the previous time-window and run the physics simulation engine one-step forward using the mapped controller actions
        self._advance(write_data)
        # (3) receive the new state (partial observation) from the physics simulation engine
        obs = self._get_observation()
        # (4) receive the instantaneous reward signal for the controller actions
        reward = self._get_reward()
        # (5) check if the physics simulation engine is still running
        terminated = not self._interface.is_coupling_ongoing()
        # (6) if the physics simulation engine has reached the end of episode time, then finalize coupling and prepare for a reset
        if terminated:
            self._interface.finalize()
            del self._interface
            logger.info("preCICE finalized and its object deleted ...\n")
            self._solver = self._finalize_subprocess(self._solver)
            self._interface = None
            self._solver_full_reset = False
            self._is_reset = False

        logger.debug(f"Step {self._env_dir} - end")

        return obs, reward, terminated, False, {}

    def close(self) -> None:
        """Close the environment by switching off the coupling and releasing all resources used by preCICE and the physics simulation engine."""
        self._finalize()

    def _set_mesh_data(self) -> None:
        """Collect name and type of boundary variables need to be communicated between the controller and the physics simulation engine."""
        scalar_variables, vector_variables, mesh_list, controller = get_mesh_data(
            self._precice_cfg
        )

        for mesh_name in mesh_list:
            if "controller" in mesh_name.lower():
                controller["mesh_name"] = mesh_name
                break

        self._controller = controller
        self._controller_mesh = self._controller["mesh_name"]
        self._scalar_variables = scalar_variables
        self._vector_variables = vector_variables
        self._read_var_list = self._controller[self._controller_mesh]["read"]
        self._write_var_list = self._controller[self._controller_mesh]["write"]

    def _set_precice_vectices(self, actuator_coords: List[np.array]) -> None:
        """Receive mesh coordinates of the controlled boundaries (actuators) of the physics simulation engine.

        Args:
            actuator_coords (List[np.array]): mesh coordinates of the controlled boundaries (actuators). These coordinates can be either
            face centres or point-based vertices of the actuator patches depending on the preCICE coupling setting.
        """
        assert actuator_coords, "actuator coords is empty!"
        self._vertex_coords_np = np.array(
            [item for sublist in actuator_coords for item in sublist]
        )
        self._precice_mesh_defined = True

    # preCICE related methods:
    def _init_precice(self) -> None:
        """Couple the physics simulation engine and the controller via preCICE, and runs the physics simulation engine one-step forward."""
        self._time_window = 0
        self._mesh_id = {}
        self._vertex_coords = {}
        self._vertex_ids = {}
        self._read_ids = {}
        self._write_ids = {}
        mesh_name = self._controller["mesh_name"]

        assert self._interface is None, "preCICE-interface re-initialisation attempt!"
        self._interface = precice.Interface("Controller", self._precice_cfg, 0, 1)

        # (1) set spatial mesh coupling data
        mesh_id = self._interface.get_mesh_id(mesh_name)
        self._mesh_id[mesh_name] = mesh_id
        vertex_ids = self._interface.set_mesh_vertices(mesh_id, self._vertex_coords_np)
        self._vertex_ids[mesh_name] = vertex_ids
        self._vertex_coords[mesh_name] = self._vertex_coords_np

        # (2) establish connection with the physics simulation engine
        self._dt = self._interface.initialize()
        self._t = self._dt

        # (3) set read/write coupling data
        mesh_name = self._controller["mesh_name"]
        for read_var in self._read_var_list:
            self._read_ids[read_var] = self._interface.get_data_id(
                read_var, self._mesh_id[mesh_name]
            )
        for write_var in self._write_var_list:
            self._write_ids[write_var] = self._interface.get_data_id(
                write_var, self._mesh_id[mesh_name]
            )
        if self._interface.is_action_required(action_write_initial_data()):
            self._interface.mark_action_fulfilled(action_write_initial_data())

        # (4) start the first time-window by taking an uncontrolled time-step forward
        self._interface.initialize_data()
        self._is_reset = True

    def _advance(self, write_data: List[str]) -> None:
        """Communicate boundary field values (obtained from mapping the controller action) with the physics simulation engine, and advances its dynamics one step forwards in time.

        Args:
            write_data (List[str]): list of variable names that their values need to be communicated with the physics simulation engine via preCICE.
        """
        self._write(write_data)

        if self._interface.is_action_required(action_write_iteration_checkpoint()):
            while True:
                self._interface.mark_action_fulfilled(
                    action_write_iteration_checkpoint()
                )
                self._dt = self._interface.advance(self._dt)
                self._interface.mark_action_fulfilled(
                    action_read_iteration_checkpoint()
                )

                if self._interface.is_time_window_complete():
                    break
        else:
            self._dt = self._interface.advance(self._dt)

        # increase the time before reading the probes/forces for internal consistency checks
        if self._interface.is_time_window_complete():
            self._time_window += 1

        if self._interface.is_coupling_ongoing():
            self._t += self._dt

        # dummy advance to finalize time-window and coupling status
        if (
            math.isclose(self._t, self._episode_end_time)
            and self._interface.is_coupling_ongoing()
        ):
            if self._interface.is_action_required(action_write_iteration_checkpoint()):
                while True:
                    self._interface.mark_action_fulfilled(
                        action_write_iteration_checkpoint()
                    )
                    self._interface.advance(self._dt)
                    self._interface.mark_action_fulfilled(
                        action_read_iteration_checkpoint()
                    )

                    if self._interface.is_time_window_complete():
                        break
            else:
                self._interface.advance(self._dt)

    def _write(self, write_data: List[str]) -> None:
        """Write boundary field values (obtained from mapping the controller action) to preCICE buffer.

        Args:
            write_data (List[str]): list of variable names that their values need to be communicated with the physics simulation engine via preCICE.
        """
        for write_var in self._write_var_list:
            if write_var in self._vector_variables:
                self._interface.write_block_vector_data(
                    self._write_ids[write_var],
                    self._vertex_ids[self._controller_mesh],
                    write_data[write_var],
                )
            elif write_var in self._scalar_variables:
                self._interface.write_block_scalar_data(
                    self._write_ids[write_var],
                    self._vertex_ids[self._controller_mesh],
                    write_data[write_var],
                )
            else:
                raise Exception(f"Invalid variable type: {write_var}")

    def _launch_subprocess(self, cmd: str):
        r"""Pre-run, reset, or run the physics simulation engine as a subprocess.

        Args:
            cmd (str): 'reset_solvers', 'prerun_solvers', or 'run_solvers'
        """
        assert cmd in [
            "reset_solvers",
            "prerun_solvers",
            "run_solvers",
        ], "Invalid command name"

        subproc_env = {
            key: variable for key, variable in os.environ.items() if "MPI" not in key
        }

        if cmd == "reset_solvers":
            for solver in self._solver_list:
                try:
                    completed_process = subprocess.run(
                        [f"./{self._reset_script}"],
                        shell=True,
                        env=subproc_env,
                        cwd=f"{self._env_dir}/{solver}",
                    )
                except Exception as err:
                    logger.error(
                        f'Failed to run {cmd} - {self._reset_script} from the folder f"{self._env_dir}/{solver}"'
                    )
                    raise err

            if completed_process.returncode != 0:
                raise Exception(f"Subprocess was not successful - {completed_process}")

        elif cmd == "prerun_solvers":
            for solver in self._solver_list:
                try:
                    completed_process = subprocess.run(
                        [f"./{self._prerun_script}"],
                        shell=True,
                        env=subproc_env,
                        cwd=f"{self._env_dir}/{solver}",
                    )
                except Exception as err:
                    logger.error(
                        f'Failed to run {cmd} - {self._prerun_script} from the folder f"{self._env_dir}/{solver}"'
                    )
                    raise err

            if completed_process.returncode != 0:
                raise Exception(f"Subprocess was not successful - {completed_process}")

        elif cmd == "run_solvers":
            subproc = []
            for solver in self._solver_list:
                subproc.append(
                    subprocess.Popen(
                        [f"./{self._run_script}"],
                        shell=True,
                        env=subproc_env,
                        cwd=f"{self._env_dir}/{solver}",
                    )
                )
            return subproc

    def _check_subprocess_exists(self, subproc_list: List) -> None:
        """Check if the physics simulation engine is successfully launched as a subprocess.

        Args:
            subproc_list (List): list of launched subprocesses.
            The size of the list is equal to the number of numerical solvers within the physics simulation engine.
        """
        for subproc, solver in zip(subproc_list, self._solver_list):
            # check if the spawning process exists
            if not psutil.pid_exists(subproc.pid):
                raise Exception(f'Failed subprocess - f"{self._env_dir}/{solver}"')

    def _finalize_subprocess(self, subproc_list: List) -> None:
        """Finalise the subprocess of the physics simulation engine upon closing the environment.

        Args:
            subproc_list (List): list of launched subprocesses.
            The size of the list is equal to the number of numerical solvers within the physics simulation engine.
        """
        for subproc, solver in zip(subproc_list, self._solver_list):
            if subproc and psutil.pid_exists(subproc.pid):
                if psutil.Process(subproc.pid).status() != psutil.STATUS_ZOMBIE:
                    logger.info(
                        "Subprocess status is not zombie - waiting to finish ..."
                    )
                    exit_signal = subproc.wait()
                else:
                    logger.info("Subprocess status is zombie - cleaning up ...")
                    exit_signal = subproc.poll()
                # check the subprocess exit signal
                if exit_signal != 0:
                    raise Exception(
                        f'Subprocess failed to complete its shell command - f"{self._env_dir}/{solver}"'
                    )
                logger.info(
                    f'Subprocess successfully completed its shell command: f"{self._env_dir}/{solver}"'
                )

    def _dummy_episode(self):
        """Run the physics simulation engine for a dummy episode to end the coupling gracefully."""
        dummy_action = 0.0 * self.action_space.sample()
        done = False
        while not done:
            _, _, done, _, _ = self.step(dummy_action)

    def _finalize(self) -> None:
        """Wrap del method."""
        self.__del__()

    def __del__(self):
        """Close all external resources, and if preCICE is still on, gracefully ends the coupling."""
        self._close_external_resources()
        if self._interface is not None:
            try:
                self._dummy_episode()
            except Exception as err:
                logger.error(f"Unsuccessful termination attempt - {err}")
                raise err

    @abstractmethod
    def _get_action(self, action: ActType, write_var_list: List) -> dict:
        """Map actions received from the controller into appropriate boundary fields to be communicated with the physics simulation engine.

        Args:
            action (ActType): an action provided by the controller to update the environment state.
            write_var_list (List): list of boundary field names to be communicated with the physics simulation engine via preCICE.

        Returns:
            dict: a dictionary containing boundary fields with their appropriate mapped values.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self) -> ObsType:
        r"""Receive partial observation information from the the physics simulation engine to be fed into the controller.

        Returns:
            ObsType: an element of the environment's :attr:`observation_space`.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self) -> float:
        """Receive the instantaneous reward signal (a scalar value) achieved as a result of taking an action in the physics simulation engine.

        Returns:
            float: an instantaneous reward signal.
        """
        raise NotImplementedError

    @abstractmethod
    def _close_external_resources(self) -> None:
        """Close external resources used by the physics simulation engine."""
        pass
