import random
import os
import numpy as np
import xmltodict
import pprint
import copy
import re


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


def parse_probe_lines(line_string):
    if line_string[0] == "#":
        print(f"comment line: {line_string}")
        is_comment = True
        return is_comment, None, None, None

    is_comment = False
    numeric_const_pattern = r"""
        [-+]? # optional sign
        (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
        )
        # followed by optional exponent part if desired
        (?: [Ee] [+-]? \d+ ) ?
    """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    float_list = rx.findall(line_string)
    float_list = [float(x) for x in float_list]

    if line_string.count("(") > 0:
        print("vector variables")
        num_probes = line_string.count("(")
        assert num_probes == line_string.count(")"), f'corrupt file, number of ( and ) should be equal:" \
            "{line_string.count(")")}, {line_string.count(")")}'
        assert (len(float_list) - 1) % num_probes == 0, f'corrupt file, each probe should have the same number of components, {len(float_list)}, {num_probes}'
    else:
        num_probes = len(float_list) - 1
    # comment or not, time idx, number of probes, probe values
    return is_comment, float_list[0], num_probes, float_list[1:]


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
