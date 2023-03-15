import os
from time import sleep

# TODO: use constant file for all constants
SLEEP_TIME = 0.5


def _make_env_dir(env_dir, solver_list):
    os.system(f'rm -rf {os.path.join(os.getcwd(), env_dir)}')
    for solver in solver_list:
        try:
            os.makedirs(os.path.join(os.getcwd(), env_dir, solver))
            os.system(f'cp -rs {os.path.join(os.getcwd(), solver)} {env_dir}')
        except Exception as e:
            raise Exception(f'Failed to create symbolic links to foam case files: {e}')
    sleep(SLEEP_TIME)


def open_file(file_name, max_wait_time=1000000):
    # wait till the file is available
    max_attempts = int(max_wait_time / 1e-6)
    acceess_counter = 0
    while True:
        try:
            file_object = open(file_name, 'r')
            break
        except IOError:
            acceess_counter += 1
            if acceess_counter < max_attempts:
                continue
            else:
                # break after trying max_attempts
                raise IOError(f'Could not access {file_name} after {max_attempts} attempts')
    return file_object
