import os
from time import sleep
from datetime import datetime
from os.path import join
import logging
from typing import Tuple, Optional, List
import json

from gymprecice.utils.constants import SLEEP_TIME, MAX_ACCESS_WAIT_TIME
from gymprecice.utils.xmlutils import _replace_keyword

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def make_env_dir(env_dir: str = None, solver_list: list = None) -> None:
    """Create a directory with all necessary solver and config files to represent a full training environment.

    Args:
        env_dir (str): envirenment directory path
        solver_list (list): list of solvers reside in the environment.
    """
    os.system(f"rm -rf {os.path.join(os.getcwd(), env_dir)}")
    for solver in solver_list:
        solver_case_dir = os.path.join(os.getcwd(), solver)
        try:
            if os.path.isdir(solver_case_dir):
                os.makedirs(os.path.join(os.getcwd(), env_dir, solver))
                os.system(f"cp -rs {solver_case_dir} {env_dir}")
            else:
                raise OSError
        except Exception as err:
            logger.error("Failed to create symbolic links to solver files")
            raise err
    sleep(SLEEP_TIME)


def open_file(file: str = None):
    """Open dynamic files."""
    max_attempts = int(MAX_ACCESS_WAIT_TIME / 1e-6)
    acceess_counter = 0
    while True:
        try:
            file_object = open(file)
            break
        except IOError:
            acceess_counter += 1
            if acceess_counter < max_attempts:
                continue
            else:
                # break after trying max_attempts
                raise IOError(f"Could not access {file} after {max_attempts} attempts")
    return file_object


def make_result_dir() -> dict:
    """Create a time-stamped result directory.

    Note:
        "precice-config.xml" is the precice configuration file that should be located in "physics-simulation-engine" directory of your problem case.\n
        "gymprecice-config.json" is the environment configuration file that should be located in "physics-simulation-engine" directory of your problem case.
        
        "gymprecice-config.json" has the following format: \n
        
        {
            "environment": {
                "name": "",
                "result_save_path": "",  // This keyword is optional
            },
            "solvers": {
                "name": [],
                "reset_script": "",
                "run_script": "",
            },
            "actuators": {
                "name": []
            }
        }
    """
    precice_config_name = "precice-config.xml"
    gymprecice_config_name = "gymprecice-config.json"

    sim_engine =  join(os.getcwd(), "physics-simulation-engine")
    precice_config = join(sim_engine, precice_config_name)
    gymprecice_config =  join(sim_engine, gymprecice_config_name)

    with open(gymprecice_config) as config_file:
        content = config_file.read()
    options = json.loads(content)
    options.update({"precice":{"config_file": precice_config_name}})

    result_path = options["environment"].get("results_path", os.getcwd())
    env_name = options["environment"]["name"]
    solver_names = options["solvers"]["name"]
    solver_dirs = [join(sim_engine, solver) for solver in solver_names]
    
    time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_dir_name = f"{env_name}_controller_training_{time_str}"
    run_dir = join(result_path, "gymprecice-run", run_dir_name)

    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as err:
        logger.error(f"Failed to create run directory")
        raise err

    try:
        for solver_dir in solver_dirs:
            os.system(f"cp -r {solver_dir} {run_dir}")
    except Exception as err:
        logger.error(f"Failed to copy base case to run direrctory")
        raise err

    try:
        os.system(f"cp {precice_config} {run_dir}")
    except Exception as err:
        logger.error(f"Failed to copy precice config file to run dir")
        raise err

    os.chdir(str(run_dir))

    keyword = "exchange-directory"
    keyword_value = f"{run_dir}/precice-{keyword}"
    _replace_keyword(
        precice_config_name,
        keyword,
        keyword_value,
        place_counter_postfix=True,
    )

    return options