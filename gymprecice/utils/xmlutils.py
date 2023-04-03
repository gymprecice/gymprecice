import xmltodict
import pprint
import copy

verbose_mode = False

def _load_file(filename):
    content = None
    with open(filename) as filehandle:
        content = filehandle.readlines()
    content = '\n'.join(content)
    return content


def get_episode_end_time(filename):
    content = _load_file(filename)
    xml_tree = xmltodict.parse(content, process_namespaces=False, dict_constructor=dict)
    solver_interface = xml_tree['precice-configuration']['solver-interface']

    max_time = None
    for key in solver_interface.keys():
        if key.rpartition(':')[0].lower() == "coupling-scheme":
            max_time = solver_interface[key]['max-time']['@value']
            break
    assert max_time is not None, f"Error: could not find max-time keyword in {filename}"

    return float(max_time)


def get_mesh_data(filename):
    content = _load_file(filename)
    xml_tree = xmltodict.parse(content, process_namespaces=False, dict_constructor=dict)
    if verbose_mode:
        pprint.pprint(xml_tree, sort_dicts=False, width=120)

    solver_interface = xml_tree['precice-configuration']['solver-interface']
    mesh_list = []
    if 'mesh' in solver_interface.keys():
        assert type(solver_interface['mesh']) == \
        list, "This version does not support single-mesh coupling. Please provide mesh for all participants"

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
            controller['read-data'] = [controller['read-data']]
        for item_ in controller['read-data']:
            if item_['@mesh'] not in controller_dict.keys():
                controller_dict[item_['@mesh']] = {"read": [], "write": []}
            controller_dict[item_['@mesh']]["read"].append(item_['@name'])

    if 'write-data' in controller.keys():
        if isinstance(controller['write-data'], dict):
            controller['write-data'] = [controller['write-data']]
        for item_ in controller['write-data']:
            if item_['@mesh'] not in controller_dict.keys():
                controller_dict[item_['@mesh']] = {"read": [], "write": []}
            controller_dict[item_['@mesh']]["write"].append(item_['@name'])

    return scaler_variables, vector_variables, mesh_list, controller_dict