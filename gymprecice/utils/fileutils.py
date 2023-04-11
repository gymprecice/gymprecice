import os
from time import sleep
from datetime import datetime
import fileinput
import sys
from os.path import join
import logging

from gymprecice.utils.constants import SLEEP_TIME, MAX_ACCESS_WAIT_TIME

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def _replace_line(
    file_name,
    keyword,
    keyword_value="",
    assignment_symbol="=",
    end_line_symbol="",
    set_counter_postfix=False,
):
    replacement_cnt = 0
    fin = fileinput.input(file_name, inplace=True)
    new_line = ""
    for line in fin:
        replaced = False
        if keyword in line and not replaced:
            if set_counter_postfix:
                new_line = f"{line.partition(keyword)[0]}{keyword}{assignment_symbol}{keyword_value}-{replacement_cnt}{end_line_symbol}"
            else:
                new_line = f"{line.partition(keyword)[0]}{keyword}{assignment_symbol}{keyword_value}{end_line_symbol}"
            new_line = new_line + "\n" if not new_line.endswith("\n") else new_line
            line = new_line
            replaced = True
            replacement_cnt += 1
        sys.stdout.write(line)
    fin.close()


def make_env_dir(env_dir, solver_list):
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


def open_file(file_name):
    # wait till the file is available
    max_attempts = int(MAX_ACCESS_WAIT_TIME / 1e-6)
    acceess_counter = 0
    while True:
        try:
            file_object = open(file_name)
            break
        except IOError:
            acceess_counter += 1
            if acceess_counter < max_attempts:
                continue
            else:
                # break after trying max_attempts
                raise IOError(
                    f"Could not access {file_name} after {max_attempts} attempts"
                )
    return file_object


def make_result_dir(options):
    env_name = options["environment"]["name"]
    env_source_path = options["environment"]["src"]
    result_path = options["environment"].get("result_save_path", os.getcwd())

    solver_names = options["solvers"]["name"]
    precice_config_file_name = options["precice"]["precice_config_file_name"]
    solver_dirs = [join(env_source_path, solver) for solver in solver_names]
    precice_config_file = join(env_source_path, precice_config_file_name)
    time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_dir_name = f"{env_name}_controller_training_{time_str}"
    run_dir = join(result_path, "gymprecice-run", run_dir_name)

    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as err:
        logger.error(f"Failed to create run directory")
        raise err

    # copy base case to run dir
    try:
        for solver_dir in solver_dirs:
            os.system(f"cp -r {solver_dir} {run_dir}")
    except Exception as err:
        logger.error(f"Failed to copy base case to run direrctory")
        raise err

    # copy precice config file to run dir
    try:
        os.system(f"cp {precice_config_file} {run_dir}")
    except Exception as err:
        logger.error(f"Failed to copy precice config file to run dir")
        raise err

    os.chdir(str(run_dir))

    keyword = "exchange-directory"
    keyword_value = f'"{run_dir}/precice-{keyword}'
    _replace_line(
        precice_config_file_name,
        keyword,
        keyword_value,
        end_line_symbol='" />',
        set_counter_postfix=True,
    )
