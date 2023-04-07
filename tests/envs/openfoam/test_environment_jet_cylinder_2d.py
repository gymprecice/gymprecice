import pytest
import requests

from os import chdir, path, getcwd, remove
import numpy as np
import math

from gymprecice.core import Adapter
from gymprecice.envs.openfoam.jet_cylinder_2d.environment import JetCylinder2DEnv
from gymprecice.envs.openfoam.jet_cylinder_2d import environment_config
from gymprecice.utils.fileutils import make_result_dir, _replace_line


@pytest.fixture
def test_case(tmpdir):
    test_dir = tmpdir.mkdir("test")
    test_env_dir = test_dir.mkdir("test-jet-cylinder-env")
    chdir(test_env_dir)


class TestJetCylinder2D(object):
    def test_base(self):
        assert JetCylinder2DEnv.__base__.__name__ == Adapter.__name__

    def test_setters(self, test_case):
        make_result_dir(environment_config)
        env = JetCylinder2DEnv(environment_config)
        env.n_probes = 10
        env.n_forces = 4
        env.min_jet_rate = -1
        env.max_jet_rate = 1

        check = {
            "n_of_probes": env._observation_info["n_probes"] == env.n_probes,
            "n_of_forces": env._reward_info["n_forces"] == env.n_forces,
            "action_space": (env.action_space.high == env.max_jet_rate \
                             and env.action_space.low == env.min_jet_rate),
            "obs_space":  env.observation_space.shape == (env.n_probes, ),
        }
        assert all(check.values())
        env.close()

    @pytest.mark.parametrize(
        "input, expected",
        [(0, [f'/postProcessing/probes/0/p',
              f'/postProcessing/forceCoeffs/0/coefficient.dat', False]),
        (0.0, [f'/postProcessing/probes/0/p',
               f'/postProcessing/forceCoeffs/0/coefficient.dat', False]),
        (0.25, [f'/postProcessing/probes/0.25/p',
                f'/postProcessing/forceCoeffs/0.25/coefficient.dat', True]),]
    )
    def test_latest_time(self, test_case, input, expected):
        make_result_dir(environment_config)
        env = JetCylinder2DEnv(environment_config)
        env.latest_available_sim_time = input

        check = {
            "obs_time_dir": env._observation_info["file_path"] == expected[0],
            "reward_time_dir": env._reward_info["file_path"] == expected[1],
            "prerun_data_required": env._prerun_data_required == expected[2]
        }
        assert all(check.values())
        env.close()
    
    @pytest.mark.parametrize(
        "input, expected",
        [(0.0001, [np.ndarray, 151, 0.05]),
         (0.0005, [np.ndarray, 151, 0.05]),
         (0.0006, [np.ndarray, 151, 0.05]),
         (0.001, [np.ndarray, 151, 0.05]),]
    )
    def test_reset(self, test_case, input, expected):
        make_result_dir(environment_config)
        _replace_line(path.join(getcwd(), "precice-config.xml"),
                      keyword="max-time value", keyword_value=f'\"{input}\"', 
                      end_line_symbol=" />")
        
        env = JetCylinder2DEnv(environment_config)
        env.n_probes = expected[1]
        obs, _ = env.reset()

        check = {
            "type": isinstance(obs, expected[0]),
            "size": obs.shape[0] == expected[1],
            "value": math.isclose(round(obs.mean(), 2), expected[2])
        }
        assert all(check.values())
        env.close()

    def test_invalid_step_order(self, test_case):
        make_result_dir(environment_config)
        env = JetCylinder2DEnv(environment_config)
        
        requests.get.side_effect = Exception
        with pytest.raises(Exception):
            env.step(env.action_space.sample())
        env.close()

    def test_step(self, test_case):
        make_result_dir(environment_config)
        max_time = 0.0015  # test for two steps
        action_interval = 1
        n_probes = 151
        _replace_line(path.join(getcwd(), "precice-config.xml"),
                        keyword="max-time value", keyword_value=f'\"{max_time}\"', 
                        end_line_symbol=" />")
        
        env = JetCylinder2DEnv(environment_config)
        env.n_probes = n_probes
        env.action_interval = action_interval
        env.reset()
        
        obs_step0, reward_step0, terminated_step0, truncated_step0, _ = env.step(env.action_space.sample())
        obs_step1, reward_step1, terminated_step1, truncated_step1, _ = env.step(env.action_space.sample())

        check = {
            "obs_size_step0": obs_step0.shape == (n_probes,),
            "obs_size_step1": obs_step1.shape == (n_probes,),
            "obs_type_step0": isinstance(obs_step0, np.ndarray),
            "obs_type_step1": isinstance(obs_step1, np.ndarray),
            "reward_type0": isinstance(reward_step0, float),
            "reward_type1": isinstance(reward_step1, float),
            "terminated_step0": terminated_step0 == False,
            "terminated_step1": terminated_step1 == True,
            "truncated_step0": truncated_step0 == False,
            "truncated_step1":  truncated_step1 == False,
        }
        assert all(check.values())
        env.close()

    def test_get_action(self, test_case):
        make_result_dir(environment_config)
        input= np.array([-1.0, 1.0])

        max_time = 0.0
        _replace_line(path.join(getcwd(), "precice-config.xml"),
                        keyword="max-time value", keyword_value=f'\"{max_time}\"', 
                        end_line_symbol=" />")
        
        env = JetCylinder2DEnv(environment_config)
        
        output0 = env._action_to_patch_field(input[0])
        output1 = env._action_to_patch_field(input[1])

        assert np.array_equal(output0, np.negative(output1))
        env.close()

    def test_get_reward(self, test_case):
        make_result_dir(environment_config)
        latest_available_sim_time = 0
        reward_average_time_window = 0.0005
        n_forces = 3
        max_time = 0.0005
        _replace_line(path.join(getcwd(), "precice-config.xml"),
                        keyword="max-time value", keyword_value=f'\"{max_time}\"', 
                        end_line_symbol=" />")

        path_to_forces_dir = path.join(
            getcwd(), f"fluid-openfoam/postProcessing/forceCoeffs/{latest_available_sim_time}/")
        remove(path.join(path_to_forces_dir, "coefficient.dat"))
        
        input = \
        '''# Time    Cd   Cs   Cl 
            0.0005  1.0  0    2.0     
        '''
        with open(path.join(path_to_forces_dir, "coefficient.dat"), "w") as file:
            file.write(input) 

        expected = 3.205 - 1 - 0.2 * 2
        
        env = JetCylinder2DEnv(environment_config)
        env.n_forces = n_forces
        env.latest_available_sim_time = latest_available_sim_time
        env.reward_average_time_window = reward_average_time_window
        env.reset()

        output = env._forces_to_reward()

        assert math.isclose(output, expected)
        env.close()
    