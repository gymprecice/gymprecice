import pytest
from pytest_mock import mocker

import os
from shutil import rmtree
import numpy as np
import math

from gymprecice.envs.openfoam.jet_cylinder_2d.environment import JetCylinder2DEnv
from gymprecice.envs.openfoam.jet_cylinder_2d import environment_config

from tests.envs.mocked_core import Adapter
JetCylinder2DEnv.__bases__ = (Adapter, )

@pytest.fixture(autouse=True)
def testdir(tmpdir):
    test_env_dir = tmpdir.mkdir("test-jet-cylinder-env")
    yield os.chdir(test_env_dir)
    rmtree(test_env_dir)

@pytest.fixture
def mock_env_helpers(mocker):
    mocker.patch("gymprecice.envs.openfoam.jet_cylinder_2d.environment.get_interface_patches",
                  return_value=[])
    mocker.patch("gymprecice.envs.openfoam.jet_cylinder_2d.environment.get_patch_geometry",
                  return_value={})

@pytest.fixture
def mock_adapter(mocker):
    Adapter._set_precice_vectices = mocker.MagicMock()


class TestJetCylinder2D:
    def test_base(self, testdir):
        assert JetCylinder2DEnv.__base__.__name__ == Adapter.__name__

    def test_setters(self, testdir, mock_env_helpers, mock_adapter):
        n_probes = 10
        n_forces = 4
        min_jet_rate = -1
        max_jet_rate = 1
        env = JetCylinder2DEnv(environment_config)
        env.n_probes =  n_probes
        env.n_forces = n_forces
        env.min_jet_rate = min_jet_rate
        env.max_jet_rate = max_jet_rate

        check = {
            "n_of_probes": env._observation_info["n_probes"] == n_probes,
            "n_of_forces": env._reward_info["n_forces"] == n_forces,
            "action_space": (env.action_space.high == max_jet_rate and env.action_space.low == min_jet_rate),
            "obs_space":  env.observation_space.shape == (n_probes, ),
        }
        assert all(check.values())

    @pytest.mark.parametrize(
        "input, expected",
        [(0, [f'/postProcessing/probes/0/p',
              f'/postProcessing/forceCoeffs/0/coefficient.dat', False]),
        (0.0, [f'/postProcessing/probes/0/p',
               f'/postProcessing/forceCoeffs/0/coefficient.dat', False]),
        (0.25, [f'/postProcessing/probes/0.25/p',
                f'/postProcessing/forceCoeffs/0.25/coefficient.dat', True]),]
    )
    def test_latest_time(self, testdir, mock_env_helpers, mock_adapter, input, expected):
        env = JetCylinder2DEnv(environment_config)
        env.latest_available_sim_time = input

        check = {
            "obs_time_dir": env._observation_info["file_path"] == expected[0],
            "reward_time_dir": env._reward_info["file_path"] == expected[1],
            "prerun_data_required": env._prerun_data_required == expected[2]
        }
        assert all(check.values())
    
    def test_get_observation(self, testdir, mock_env_helpers, mock_adapter, mocker):
        latest_available_sim_time = 0.335
        n_probes = 5
        
        path_to_probes_dir = os.path.join(
            os.getcwd(), f"postProcessing/probes/{latest_available_sim_time}/")
        os.makedirs(path_to_probes_dir)
        
        input = \
        '''# Time       p0      p1      p2      p3      p4 
            0.335       1.0     2.0     3.0     4.0     5.0     
        '''
        with open(os.path.join(path_to_probes_dir, "p"), "w") as file:
            file.write(input)

        env = JetCylinder2DEnv(environment_config)
        env.n_probes = n_probes
        env.latest_available_sim_time = latest_available_sim_time
        env._openfoam_solver_path = os.getcwd()

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        output = env._probes_to_observation()

        assert np.array_equal(output, expected)
    
    def test_get_action(self, testdir, mock_adapter, mocker):
        mocker.patch("gymprecice.envs.openfoam.jet_cylinder_2d.environment.get_interface_patches",
                     return_value=["jet1", "jet2"])

        env_source_path = environment_config['environment']['src']
        solver_names = environment_config['solvers']['name']
        solver_dir = [os.path.join(env_source_path, solver) for solver in solver_names][0]
        os.makedirs(f'{os.getcwd()}/{solver_names[0]}/constant')
        os.system(f'cp -r {solver_dir}/constant {os.getcwd()}/{solver_names[0]}')

        input= np.array([-1.0, 1.0])
        
        env = JetCylinder2DEnv(environment_config)        
        output0 = env._action_to_patch_field(input[0])
        output1 = env._action_to_patch_field(input[1])

        assert np.array_equal(output0, np.negative(output1))
    
    def test_get_reward(self, testdir, mock_env_helpers, mock_adapter, mocker):
        latest_available_sim_time = 0.335
        reward_average_time_window = 1
        n_forces = 3

        path_to_forces_dir_0 = os.path.join(
            os.getcwd(), f"postProcessing/forceCoeffs/0/")
        os.makedirs(path_to_forces_dir_0)

        path_to_forces_dir_1 = os.path.join(
            os.getcwd(), f"postProcessing/forceCoeffs/{latest_available_sim_time}/")
        os.makedirs(path_to_forces_dir_1)
        
        input = \
        '''# Time    Cd   Cs   Cl 
            0.335  1.0  0    2.0     
        '''
        with open(os.path.join(path_to_forces_dir_0, "coefficient.dat"), "w") as file:
            file.write(input)
        with open(os.path.join(path_to_forces_dir_1, "coefficient.dat"), "w") as file:
            file.write(input)

        env = JetCylinder2DEnv(environment_config)
        env.n_forces = n_forces
        env.latest_available_sim_time = latest_available_sim_time
        env.reward_average_time_window = reward_average_time_window
        env._openfoam_solver_path = os.getcwd()

        expected = 3.205 - 1 - 0.2 * 2
        output = env._forces_to_reward()

        assert math.isclose(output, expected)
