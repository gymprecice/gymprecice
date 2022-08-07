from sqlitedict import SqliteDict
from utils import fix_randseeds
import subprocess
import time
import copy

if __name__ == '__main__':
    rand_seed = 12345
    fix_randseeds(rand_seed)
    foamserver = SqliteDict("foamserver.sqlite", autocommit=True)

    while True:
        # print(list(foamserver.keys()))
        for key_ in foamserver.keys():
            try:
                process_dict = copy.deepcopy(foamserver[key_])
            except Exception as e:
                print(f'problem with accessing a key: {e}')
                pass
            assert 'command' in process_dict.keys(), 'command_str is not in keys'
            assert 'command_str' in process_dict.keys(), 'command_str is not in keys'
            assert 'run' in process_dict.keys(), 'run is not in keys'
            assert 'process_pid' in process_dict.keys(), 'process_pid is not in keys'

            cmd_str = process_dict['command_str']
            process_pid = process_dict['process_pid']
            if process_dict['run'] is False:
                if process_dict['command'] == 'popen':
                    subproc = subprocess.Popen(cmd_str, shell=True, cwd=key_)
                    process_dict['process_pid'] = subproc.pid
                    process_dict['run'] = True
                    foamserver[key_] = process_dict
                    print(f'popen ====== from foamserver resulting in {subproc.pid}')
                    print(key_, cmd_str, subproc.pid)
                elif process_dict['command'] == 'run':
                    subproc = subprocess.run(cmd_str, shell=True, cwd=key_)
                    process_dict['process_pid'] = 99
                    process_dict['run'] = True
                    foamserver[key_] = process_dict
                    print(f'run ====== from foamserver resulting in {subproc}')
                    print(key_, cmd_str, subproc)
                else:
                    raise Exception(f"commands are either popen or run -- received {process_dict['command']}")

        time.sleep(0.001)