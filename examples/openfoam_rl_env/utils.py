import random
import os
import numpy as np
import xmltodict
import pprint
import copy


def fix_randseeds(seed=1234):
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(e)
        pass


def load_file(foldername, filename):
    foam_text = None
    with open(foldername + filename, 'r') as filehandle:
        foam_text = filehandle.readlines()
    foam_text = '\n'.join(foam_text)
    return foam_text


def get_cfg_data(foldername, filename):

    foam_string = load_file(foldername, filename)
    # print(foam_string)

    xml_tree = xmltodict.parse(foam_string, process_namespaces=False)
    pprint.pprint(xml_tree, sort_dicts=False, width=120)

    try:
        scaler_variables = []
        for dict_ in xml_tree['precice-configuration']['solver-interface']['data:scalar']:
            print(dict_['@name'])
            scaler_variables.append(dict_['@name'])
    except Exception as e:
        pass

    try:
        vector_variables = []
        for dict_ in xml_tree['precice-configuration']['solver-interface']['data:vector']:
            print(dict_['@name'])
            vector_variables.append(dict_['@name'])
    except Exception as e:
        pass

    for item_ in xml_tree['precice-configuration']['solver-interface']['participant']:
        if 'rl-gym' in item_['@name'].lower():
            gym_data = copy.deepcopy(item_)
            print(gym_data)
            break

    mesh_variables = {}
    if 'read-data' in gym_data.keys():
        if isinstance(gym_data['read-data'], dict):
            gym_data['read-data'] = [gym_data['read-data']]

        for item_ in gym_data['read-data']:
            if item_['@mesh'] not in mesh_variables.keys():
                mesh_variables[item_['@mesh']] = {"read": [], "write": []}
            mesh_variables[item_['@mesh']]["read"].append(item_['@name'])

    if 'write-data' in gym_data.keys():
        if isinstance(gym_data['write-data'], dict):
            gym_data['write-data'] = [gym_data['write-data']]

        for item_ in gym_data['write-data']:
            if item_['@mesh'] not in mesh_variables.keys():
                mesh_variables[item_['@mesh']] = {"read": [], "write": []}
            mesh_variables[item_['@mesh']]["write"].append(item_['@name'])

    mesh_list = list(mesh_variables.keys())

    return scaler_variables, vector_variables, mesh_list, mesh_variables


if __name__ == '__main__':

    foldername = ""
    filename = "precice-config.xml"
    scaler_variables, vector_variables, mesh_list, mesh_variables = get_cfg_data(foldername, filename)
    print('========= parsing the configurations from xml file =====')
    print(scaler_variables)
    print(vector_variables)
    print(mesh_list)
    print(mesh_variables)
