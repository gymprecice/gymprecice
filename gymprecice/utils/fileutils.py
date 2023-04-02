import os
from time import sleep

from gymprecice.utils.constants import SLEEP_TIME, MAX_ACCESS_WAIT_TIME

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
