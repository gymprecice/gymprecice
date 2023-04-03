import os
from time import sleep
from datetime import datetime
import fileinput
import sys
from os.path import join

from gymprecice.utils.constants import SLEEP_TIME, MAX_ACCESS_WAIT_TIME

def _replace_keyword(file_name, find_keyword, keyword_value, keyword_end=""):
    replacement_cnt = 0
    fin = fileinput.input(file_name, inplace=True)
    for line in fin:
        replaced = False
        if find_keyword in line and not replaced:
            new = f'{line.partition(find_keyword)[0]} {find_keyword}={keyword_value}-{replacement_cnt}{keyword_end}'
            new = new + "\n" if not new.endswith("\n") else new
            line = new
            replaced = True
            replacement_cnt += 1
        sys.stdout.write(line)
    fin.close()

def make_env_dir(env_dir, solver_list):
    os.system(f'rm -rf {os.path.join(os.getcwd(), env_dir)}')
    for solver in solver_list:
        solver_case_dir = os.path.join(os.getcwd(), solver)
        try:
            if os.path.isdir(solver_case_dir):
                os.makedirs(os.path.join(os.getcwd(), env_dir, solver))
                os.system(f'cp -rs {solver_case_dir} {env_dir}')
            else:
                raise Exception
        except Exception as e:
            raise Exception(f'Failed to create symbolic links to solver files: {e}')
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
                raise IOError(f'Could not access {file_name} after {max_attempts} attempts')
    return file_object


def make_result_dir(options):
    env_name = options['environment']['name']
    env_source_path = options['environment']['src']
    result_path = options['environment'].get('result_save_path', os.getcwd()) 
    
    solver_names = options['solvers']['name']
    precice_config_file_name = options['precice']['precice_config_file_name']
    solver_dirs = [join(env_source_path, solver) for solver in solver_names]
    precice_config_file = join(env_source_path, precice_config_file_name)
    time_str = datetime.now().strftime('%d%m%Y_%H%M%S')
    run_dir_name = f'{env_name}_controller_training_{time_str}'
    run_dir = join(result_path, 'gymprecice-run', run_dir_name)

    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as e:
        raise Exception(f'failed to create run directory: {e}')

    # copy base case to run dir
    try:
        for solver_dir in solver_dirs:
            os.system(f'cp -r {solver_dir} {run_dir}')
    except Exception as e:
        raise Exception(f'Failed to copy base case to run direrctory: {e}')

    # copy precice config file to run dir
    try:
        os.system(f'cp {precice_config_file} {run_dir}')
    except Exception as e:
        raise Exception(f'Failed to copy precice config file to run dir: {e}')

    os.chdir(str(run_dir))

    keyword = "exchange-directory"
    keyword_value = f'\"{run_dir}/precice-{keyword}'
    _replace_keyword(precice_config_file_name, keyword, keyword_value, keyword_end="\"/>")


