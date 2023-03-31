import os
import xmltodict
import pprint
import copy
from os.path import join
import os
from datetime import datetime
import fileinput
import sys

verbose_mode = False


def _load_file(foldername, filename):
    foam_text = None
    if foldername and foldername[-1] != '/':
        foldername += '/'
    with open(foldername + filename, 'r') as filehandle:
        foam_text = filehandle.readlines()
    foam_text = '\n'.join(foam_text)
    return foam_text


def get_episode_end_time(filename):
    foam_string = _load_file('', filename)
    xml_tree = xmltodict.parse(foam_string, process_namespaces=False, dict_constructor=dict)
    solver_interface = xml_tree['precice-configuration']['solver-interface']

    max_time = None
    for key in solver_interface.keys():
        if key.rpartition(':')[0].lower() == "coupling-scheme":
            max_time = solver_interface[key]['max-time']['@value']
            break
    assert max_time is not None, f"Error: could not find max-time keyword in {filename}"

    return max_time


def get_mesh_data(foldername, filename):
    xml_string = _load_file(foldername, filename)
    xml_tree = xmltodict.parse(xml_string, process_namespaces=False, dict_constructor=dict)
    if verbose_mode:
        pprint.pprint(xml_tree, sort_dicts=False, width=120)

    solver_interface = xml_tree['precice-configuration']['solver-interface']
    mesh_list = []
    if 'mesh' in solver_interface.keys():
        if isinstance(solver_interface['mesh'], dict):
            solver_interface['mesh'] = [solver_interface['mesh']]
        for item_ in solver_interface['mesh']:
            mesh_list.append(item_['@name'])

    scaler_variables = []
    if 'data:scalar' in solver_interface.keys():
        if isinstance(solver_interface['data:scalar'], dict):
            solver_interface['data:scalar'] = [solver_interface['data:scalar']]
        for item_ in solver_interface['data:scalar']:
            scaler_variables.append(item_['@name'])

    vector_variables = []
    if 'data:vector' in solver_interface.keys():
        if isinstance(solver_interface['data:vector'], dict):
            solver_interface['data:vector'] = [solver_interface['data:vector']]
        for item_ in solver_interface['data:vector']:
            vector_variables.append(item_['@name'])

    # we only have one controller within our participants
    for item_ in xml_tree['precice-configuration']['solver-interface']['participant']:
        if 'controller' in item_['@name'].lower():
            controller = copy.deepcopy(item_)
            if verbose_mode:
                print(controller)
            break
    
    controller_dict = {}
    if 'read-data' in controller.keys():
        if isinstance(controller['read-data'], dict):
            controller_dict['read-data'] = [controller['read-data']]
        for item_ in controller_dict['read-data']:
            if item_['@mesh'] not in controller.keys():
                controller_dict[item_['@mesh']] = {"read": [], "write": []}
            controller_dict[item_['@mesh']]["read"].append(item_['@name'])

    if 'write-data' in controller.keys():
        if isinstance(controller['write-data'], dict):
            controller_dict['write-data'] = [controller['write-data']]
        for item_ in controller_dict['write-data']:
            if item_['@mesh'] not in controller.keys():
                controller_dict[item_['@mesh']] = {"read": [], "write": []}
            controller_dict[item_['@mesh']]["write"].append(item_['@name'])

    return scaler_variables, vector_variables, mesh_list, controller_dict


def set_training_dir(options):
    env_name = options['environment']['name']
    env_source_path = options['environment']['src']
    training_path = options['environment']['training_path']
    
    solver_names = options['solvers']['name']
    precice_config_file_name = options['precice']['precice_config_file_name']
    solver_dirs = [join(env_source_path, solver) for solver in solver_names]
    precice_config_file = join(env_source_path, precice_config_file_name)
    time_str = datetime.now().strftime('%d%m%Y_%H%M%S')
    run_dir_name = f'{env_name}_controller_training_{time_str}'
    run_dir = join(training_path, 'gymprecice-run', run_dir_name)

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

    # set precice exchange-directory if communicating via sockets
    keyword = "exchange-directory"
    replacement_cnt = 0
    fin = fileinput.input(precice_config_file_name, inplace=True)
    for line in fin:
        replaced = False
        if keyword in line and not replaced:
            new = f'{line.partition(keyword)[0]} {keyword}="{run_dir}/precice-{keyword}-{replacement_cnt}" />'
            new = new + "\n" if not new.endswith("\n") else new
            line = new
            replaced = True
            replacement_cnt += 1
        sys.stdout.write(line)
    fin.close()