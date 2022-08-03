import random
import os
from typing import OrderedDict
import numpy as np
import xmltodict
import pprint
import copy
import re
from functools import partial


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
    if foldername and foldername[-1] != '/':
        foldername += '/'
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
        print("vector variable")
        num_probes = line_string.count("(")
        assert num_probes == line_string.count(")"), f'corrupt file, number of ( and ) should be equal:" \
            "{line_string.count(")")}, {line_string.count(")")}'
        assert (len(float_list) - 1) % num_probes == 0, f'corrupt file, each probe should have the same number of components, {len(float_list)}, {num_probes}'
    else:
        num_probes = len(float_list) - 1
    # comment or not, time idx, number of probes, probe values
    return is_comment, float_list[0], num_probes, float_list[1:]


def get_xml_item(root, chain_to_root: list, n: int = 0):
    if n < len(chain_to_root):
        return get_xml_item(root[chain_to_root[n]], chain_to_root, n + 1)
    else:
        return [root] if type(root) == dict else root


def get_cfg_data(foldername, filename):
    foam_string = load_file(foldername, filename)
    xml_tree = xmltodict.parse(foam_string, process_namespaces=False, dict_constructor=dict)
    pprint.pprint(xml_tree, sort_dicts=False, width=120)

    # try:
    #     mesh_list = []
    #     xml_item = get_xml_item(xml_tree, ['precice-configuration', 'solver-interface', 'mesh'])
    #     for element in xml_item:
    #         mesh_list.append(element['@name'])
    # except Exception as e:
    #     print(e)

    # try:
    #     scaler_variables = []
    #     xml_item = get_xml_item(xml_tree, ['precice-configuration', 'solver-interface', 'data:scalar'])
    #     for element in xml_item:
    #         scaler_variables.append(element['@name'])
    # except Exception as e:
    #     print(e)

    # try:
    #     vector_variables = []
    #     xml_item = get_xml_item(xml_tree, ['precice-configuration', 'solver-interface', 'data:vector'])
    #     for element in xml_item:
    #         vector_variables.append(element['@name'])
    # except Exception as e:
    #     print(e)

    # xml_item = get_xml_item(xml_tree, ['precice-configuration', 'solver-interface', 'participant'])
    # for item_ in xml_item:
    #     if 'rl-gym' in item_['@name'].lower():
    #         gym_data = copy.deepcopy(item_)
    #         print(gym_data)
    #         break

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
    if 'data:scalar' in solver_interface.keys():
        if isinstance(solver_interface['data:vector'], dict):
            solver_interface['data:vector'] = [solver_interface['data:vector']]
        for item_ in solver_interface['data:vector']:
            vector_variables.append(item_['@name'])

    # we only have one rl-gym participant
    for item_ in xml_tree['precice-configuration']['solver-interface']['participant']:
        if 'rl-gym' in item_['@name'].lower():
            gym_participant = copy.deepcopy(item_)
            print(gym_participant)
            break

    mesh_variables = {}
    if 'read-data' in gym_participant.keys():
        if isinstance(gym_participant['read-data'], dict):
            gym_participant['read-data'] = [gym_participant['read-data']]
        for item_ in gym_participant['read-data']:
            if item_['@mesh'] not in mesh_variables.keys():
                mesh_variables[item_['@mesh']] = {"read": [], "write": []}
            mesh_variables[item_['@mesh']]["read"].append(item_['@name'])

    if 'write-data' in gym_participant.keys():
        if isinstance(gym_participant['write-data'], dict):
            gym_participant['write-data'] = [gym_participant['write-data']]

        for item_ in gym_participant['write-data']:
            if item_['@mesh'] not in mesh_variables.keys():
                mesh_variables[item_['@mesh']] = {"read": [], "write": []}
            mesh_variables[item_['@mesh']]["write"].append(item_['@name'])

    return scaler_variables, vector_variables, mesh_list, mesh_variables


def repeat_variable(variables_list, n_parallel_env):
    new_list = []
    # if we have one item make it a list
    if isinstance(variables_list, dict):
        variables_list = [variables_list]
    for var_dict in variables_list:
        var_name = var_dict['@name']
        for idx in range(n_parallel_env):
            new_list.append({'@name': var_name + f'_{idx}'})
    # print(new_list)
    return new_list


def append_variable(variables_list, idx):
    new_list = []
    # if we have one item make it a list
    if isinstance(variables_list, dict):
        variables_list = [variables_list]

    for var_dict in variables_list:
        var_name = var_dict['@name']
        new_list.append({'@name': var_name + f'_{idx}'})
    return new_list


def mod_use_mesh_list(use_mesh_list, solver_indices):
    if isinstance(use_mesh_list, dict):
        use_mesh_list = [use_mesh_list]
    new_use_mesh_list = []
    for mesh_item in use_mesh_list:
        if 'rl-gym' in mesh_item['@name'].lower():
            new_use_mesh_list.append(mesh_item)
        else:
            for idx in solver_indices:
                if '@from' in mesh_item.keys():
                    if 'rl-gym' in mesh_item['@from'].lower():
                        new_use_mesh_list.append({
                            '@name': mesh_item['@name'] + f'_{idx}',
                            '@from': mesh_item['@from'],
                        })
                    else:
                        new_use_mesh_list.append({
                            '@name': mesh_item['@name'] + f'_{idx}',
                            '@from': mesh_item['@from'] + f'_{idx}',
                        })
                # elif '@to' in mesh_item.keys():
                #     new_use_mesh_list.append({
                #         '@name': mesh_item['@name'] + f'_{idx}',
                #         '@to': mesh_item['@to'] + f'_{idx}',
                #     })
                elif '@provide' in mesh_item.keys():
                    new_use_mesh_list.append({
                        '@name': mesh_item['@name'] + f'_{idx}',
                        '@provide': mesh_item['@provide'],
                    })
                else:
                    raise Exception(f'Check the condition for mesh list: {mesh_item}')

    return new_use_mesh_list


def mod_read_write_lists(rw_data_list, solver_indices):
    if isinstance(rw_data_list, dict):
        rw_data_list = [rw_data_list]
    print(rw_data_list)
    new_rw_data_list = []

    for rw_item in rw_data_list:
        for idx in solver_indices:
            if 'rl-gym' in rw_item['@mesh'].lower():
                new_rw_data_list.append({
                    '@name': rw_item['@name'] + f'_{idx}',
                    '@mesh': rw_item['@mesh'],
                })
            else:
                new_rw_data_list.append({
                    '@name': rw_item['@name'] + f'_{idx}',
                    '@mesh': rw_item['@mesh'] + f'_{idx}',
                })
    return new_rw_data_list


def mod_mapping_lists(mapping_list, solver_indices):
    ''' n_parallel_env could be single index or '''
    if isinstance(mapping_list, dict):
        mapping_list = [mapping_list]
    new_mapping_list = []
    print(mapping_list)
    for mapping_item in mapping_list:
        for idx in solver_indices:
            new_mapping_list.append({
                '@direction': mapping_item['@direction'],
                '@from': mapping_item['@from'] + f'_{idx}',
                '@to': mapping_item['@to'] + f'_{idx}',
                '@constraint': mapping_item['@constraint'],
            })

            # if 'rl-gym' in mapping_item['@from'].lower():
            #     new_mapping_list.append({
            #         '@direction': mapping_item['@direction'],
            #         '@from': mapping_item['@from'],
            #         '@to': mapping_item['@to'] + f'_{idx}',
            #         '@constraint': mapping_item['@constraint'],
            #     })
            # elif 'rl-gym' in mapping_item['@to'].lower():
            #     new_mapping_list.append({
            #         '@direction': mapping_item['@direction'],
            #         '@from': mapping_item['@from'] + f'_{idx}',
            #         '@to': mapping_item['@to'],
            #         '@constraint': mapping_item['@constraint'],
            #     })

    return new_mapping_list


def repeat_m2n(m2n_list, n_parallel_env, parallel_folders_list):
    new_m2n_list = []
    # if we have one item make it a list
    if isinstance(m2n_list, dict):
        m2n_list = [m2n_list]

    for m2n_item in m2n_list:
        for idx in range(n_parallel_env):
            if 'rl-gym' in m2n_item['@from'].lower():
                new_m2n_list.append({
                    '@from': m2n_item['@from'],
                    '@to': m2n_item['@to'] + f'_{idx}',
                    '@exchange-directory': parallel_folders_list[idx] + "/"  # m2n_item['@exchange-directory']
                })
            elif 'rl-gym' in m2n_item['@to'].lower():
                new_m2n_list.append({
                    '@from': m2n_item['@from'] + f'_{idx}',
                    '@to': m2n_item['@to'],
                    '@exchange-directory': parallel_folders_list[idx] + "/"  # m2n_item['@exchange-directory']
                })
            else:
                print(m2n_item)
                raise Exception('undefined case for parallel parsing of m2n block')
    return new_m2n_list


def repeat_coupling(coupling_dict, n_parallel_env):
    new_coupling_list = []
    # if we have one item make it a list
    if not isinstance(coupling_dict, dict):
        raise Exception('parallel parsing: serial-explicit coupling is limited one item')
    for solver_idx in range(n_parallel_env):
        current_dict = copy.deepcopy(coupling_dict)

        participants = current_dict['participants']
        if 'rl-gym' not in participants['@first'].lower():
            participants['@first'] = participants['@first'] + f'_{solver_idx}'
        elif 'rl-gym' not in participants['@second'].lower():
            participants['@second'] = participants['@second'] + f'_{solver_idx}'
        else:
            raise Exception('parallel parsing: rl-gym have to be either the first or second')
        current_dict['participants'] = participants  # is this copy needed?

        exchange_list = current_dict['exchange']
        new_exchange_list = []
        for exchange_item in exchange_list:
            # exchange_item = copy.deepcopy(exchange_item)
            if 'rl-gym' not in exchange_item['@data'].lower():
                exchange_item['@data'] = exchange_item['@data'] + f'_{solver_idx}'
            if 'rl-gym' not in exchange_item['@mesh'].lower():
                exchange_item['@mesh'] = exchange_item['@mesh'] + f'_{solver_idx}'
            if 'rl-gym' not in exchange_item['@from'].lower():
                exchange_item['@from'] = exchange_item['@from'] + f'_{solver_idx}'
            if 'rl-gym' not in exchange_item['@to'].lower():
                exchange_item['@to'] = exchange_item['@to'] + f'_{solver_idx}'
            print(exchange_item)
            new_exchange_list.append(exchange_item)

        current_dict['exchange'] = new_exchange_list

        new_coupling_list.append(current_dict)
    return new_coupling_list


def make_parallel_config(foldername, filename, n_parallel_env, parallel_folders_list, use_mapping):
    assert n_parallel_env == len(parallel_folders_list), f'different sizes of parallel env {n_parallel_env}, {len(parallel_folders_list)}'

    foam_string = load_file(foldername, filename)

    tree_dict = xmltodict.parse(foam_string, process_namespaces=False)
    # pprint.pprint(tree_dict, sort_dicts=False, width=120)

    # print(get_cfg_data(tree_dict))

    mod_tree = copy.deepcopy(tree_dict)
    mod_tree_sub = mod_tree['precice-configuration']['solver-interface']
    # dict_keys(['@dimensions', 'data:scalar', 'mesh', 'participant', 'm2n:sockets', 'coupling-scheme:serial-implicit'])
    for key in mod_tree_sub.keys():
        if key == "@dimensions":
            pass
        elif key == "@experimental":
            pass
        elif key == "data:scalar":
            # print(mod_tree_sub[key])
            mod_tree_sub[key] = repeat_variable(mod_tree_sub[key], n_parallel_env)
        elif key == "data:vector":
            # print(mod_tree_sub[key])
            mod_tree_sub[key] = repeat_variable(mod_tree_sub[key], n_parallel_env)
        elif key == "mesh":
            # print(mod_tree_sub[key])
            # list of meshes
            new_mesh_list = []
            mesh_list = mod_tree_sub[key]
            if isinstance(mesh_list, dict):
                mesh_list = [mesh_list]

            for item in mesh_list:
                if 'rl-gym' in item['@name'].lower():
                    # repeat the variables
                    variables_list = item['use-data']
                    # print(item)
                    new_mesh_list.append({
                        '@name': item['@name'],
                        'use-data': repeat_variable(variables_list, n_parallel_env)
                    })
                    # print(new_mesh_list)
                else:
                    variables_list = item['use-data']
                    for idx in range(n_parallel_env):
                        new_mesh_list.append({
                            '@name': item['@name'] + f'_{idx}',
                            'use-data': append_variable(variables_list, idx),
                        })
                    pass
            mod_tree_sub[key] = new_mesh_list
        elif key == "participant":
            new_participant_list = []
            participant_list = mod_tree_sub[key]
            print(participant_list)
            for participant_item in participant_list:
                if 'rl-gym' in participant_item['@name'].lower():
                    print("===========")
                    print(participant_item.keys())
                    print("===========")

                    use_mesh_list = participant_item['use-mesh']
                    new_use_mesh_list = mod_use_mesh_list(use_mesh_list, range(n_parallel_env))
                    print("==== use_mesh_list")
                    print(use_mesh_list)
                    print(new_use_mesh_list)

                    write_data_list = participant_item['write-data']
                    new_write_data_list = mod_read_write_lists(write_data_list, range(n_parallel_env))
                    print("==== write_data_list")
                    print(write_data_list)
                    print(new_write_data_list)

                    read_data_list = participant_item['read-data']
                    new_read_data_list = mod_read_write_lists(read_data_list, range(n_parallel_env))
                    print("==== read_data_list")
                    print(read_data_list)
                    print(new_read_data_list)

                    if use_mapping and 'mapping:nearest-neighbor' in participant_item.keys():
                        # for now we only support mapping:nearest-neighbor
                        mapping_list = participant_item['mapping:nearest-neighbor']
                        new_mapping_list = mod_mapping_lists(mapping_list, range(n_parallel_env))
                        print("==== mapping_list")
                        print(mapping_list)
                        print(new_mapping_list)

                        new_participant_list.append({
                            '@name': participant_item['@name'],
                            'use-mesh': new_use_mesh_list,
                            'write-data': new_write_data_list,
                            'read-data': new_read_data_list,
                            'mapping:nearest-neighbor': new_mapping_list
                        })
                    else:
                        new_participant_list.append({
                            '@name': participant_item['@name'],
                            'use-mesh': new_use_mesh_list,
                            'write-data': new_write_data_list,
                            'read-data': new_read_data_list,
                        })

                else:
                    print("=========== Non RL-Gym case")
                    print(participant_item.keys())
                    print(participant_item)
                    for solver_idx in range(n_parallel_env):
                        use_mesh_list = participant_item['use-mesh']
                        new_use_mesh_list = mod_use_mesh_list(use_mesh_list, [solver_idx])
                        print("==== use_mesh_list")
                        print(use_mesh_list)
                        print(new_use_mesh_list)

                        write_data_list = participant_item['write-data']
                        new_write_data_list = mod_read_write_lists(write_data_list, [solver_idx])
                        print("==== write_data_list")
                        print(write_data_list)
                        print(new_write_data_list)

                        read_data_list = participant_item['read-data']
                        new_read_data_list = mod_read_write_lists(read_data_list, [solver_idx])
                        print("==== read_data_list")
                        print(read_data_list)
                        print(new_read_data_list)

                        if use_mapping:
                            # for now we only support mapping:nearest-neighbor
                            try:
                                mapping_list = participant_item['mapping:nearest-neighbor']
                            except Exception as e:
                                raise Exception(f'Exception {e}: parallel processing of precice xml configurations only supports mapping:nearest-neighbor for the time being')
                            new_mapping_list = mod_mapping_lists(mapping_list, [solver_idx])
                            print("==== mapping_list")
                            print(mapping_list)
                            print(new_mapping_list)

                            new_participant_list.append({
                                '@name': participant_item['@name'] + f'_{solver_idx}',
                                'use-mesh': new_use_mesh_list,
                                'write-data': new_write_data_list,
                                'read-data': new_read_data_list,
                                'mapping:nearest-neighbor': new_mapping_list
                            })
                        else:
                            new_participant_list.append({
                                '@name': participant_item['@name'] + f'_{solver_idx}',
                                'use-mesh': new_use_mesh_list,
                                'write-data': new_write_data_list,
                                'read-data': new_read_data_list,
                            })

            mod_tree_sub[key] = new_participant_list
        elif key == "m2n:sockets":
            new_m2n_list = []
            m2n_list = mod_tree_sub[key]
            print("===== m2n:sockets")
            print(m2n_list)
            mod_tree_sub[key] = repeat_m2n(m2n_list, n_parallel_env, parallel_folders_list)
        elif key == "coupling-scheme:parallel-explicit":
            coupling_dict = mod_tree_sub[key]
            print("===== coupling scheme")
            print(coupling_dict)
            mod_tree_sub[key] = repeat_coupling(coupling_dict, n_parallel_env)
        else:
            raise Exception(f'parallel xml config: unknown dictionary key: {key}')

    # Replace the serial coupling with parallel coupling --> after the iteration on the keys
    # parallel-explicit works correctly !!
    # if n_parallel_env > 1:
    #     mod_tree_sub["coupling-scheme:parallel-explicit"] = mod_tree_sub["coupling-scheme:serial-explicit"]
    #     del mod_tree_sub["coupling-scheme:serial-explicit"]

    # pprint.pprint(mod_tree_sub, sort_dicts=False, width=120)

    mod_tree['precice-configuration']['solver-interface'] = mod_tree_sub
    return mod_tree


def augment_str(match_obj, leading_string, trailing_string, idx):
    # match items that could be repeated but on different lines
    if match_obj.group() is not None:
        matched_str = str(match_obj.group())
        matched_str = matched_str.replace(leading_string, "")
        matched_str = matched_str.replace(trailing_string, "")
        matched_str = matched_str.strip()
        modified_str = leading_string + " " + matched_str + f'_{idx}' + " " + trailing_string
        print(modified_str)
        return modified_str


def augment_str2(match_obj, leading_string, trailing_string, idx):
    # match items with that could be repeated within brackets on same line or multiple lines
    if match_obj.group() is not None:
        matched_str = str(match_obj.group())
        # print(matched_str)
        matched_str = matched_str.replace(leading_string, "")
        matched_str = matched_str.replace(trailing_string, "")
        matched_str = matched_str.replace("(", "")
        matched_str = matched_str.replace(")", "")

        matched_str_list = [x.strip() + f'_{idx}' for x in re.split(r"\n|[' ']", matched_str) if len(x.strip()) > 0]
        modified_str = leading_string + " (" + " ".join(matched_str_list) + " )" + trailing_string
        print(modified_str)
        return modified_str


def parallel_precice_dict(precicedict_str, idx_):
    """ preciceDict in the OpenFoam Case need to be adapted for parallel nameing  
        very quick hack based on regex
    """
    precicedict_str = copy.deepcopy(precicedict_str)

    # find participant name --> expect a single participant in the file
    leading_string = "participant"
    trailing_string = ";"
    pattern = fr"{leading_string}\s*(.*)\s*{trailing_string}"
    precicedict_str = re.sub(
        pattern,
        partial(augment_str, leading_string=leading_string, trailing_string=trailing_string, idx=idx_), 
        precicedict_str)

    # find mesh name --> may be more than one mesh
    leading_string = "mesh"
    trailing_string = ";"
    pattern = fr"{leading_string}\s*(.*)\s*{trailing_string}"
    precicedict_str = re.sub(
        pattern,
        partial(augment_str, leading_string=leading_string, trailing_string=trailing_string, idx=idx_), 
        precicedict_str)

    # find readData names
    leading_string = "readData"
    trailing_string = ";"
    # ? for shortest match and then shortest number of repetitions
    pattern = fr"{leading_string}\s*\((\s*.*?)*?\){trailing_string}"   # fr"{leading_string}\s*\(.*\){trailing_string}"
    precicedict_str = re.sub(
        pattern,
        partial(augment_str2, leading_string=leading_string, trailing_string=trailing_string, idx=idx_),
        precicedict_str)

    # find writeData names
    leading_string = 'writeData'
    trailing_string = ";"
    # ? for shortest match and then shortest number of repetitions
    pattern = fr"{leading_string}\s*\((\s*.*?)*?\){trailing_string}"   # fr"{leading_string}\s*\(.*\){trailing_string}"
    precicedict_str = re.sub(
        pattern,
        partial(augment_str2, leading_string=leading_string, trailing_string=trailing_string, idx=idx_),
        precicedict_str)

    # clean the file from the extra newlines
    while '\n\n' in precicedict_str:
        precicedict_str = precicedict_str.replace("\n\n", "\n")

    return precicedict_str


if __name__ == '__main__':
    foldername = ""
    filename = "precice-config.xml"
    scaler_variables, vector_variables, mesh_list, mesh_variables = get_cfg_data(foldername, filename)
    print('========= parsing the configurations from xml file =====')
    print(scaler_variables)
    print(vector_variables)
    print(mesh_list)
    print(mesh_variables)

    foldername = "fluid-openfoam/system/"
    filename = 'preciceDict'
    precicedict_str = load_file(foldername, filename)
    for idx_ in range(2):
        new_string = parallel_precice_dict(precicedict_str, idx_)
        print(new_string)
        with open(filename + f'_{idx_}', 'w') as file_obj:
            file_obj.write(new_string)

    # test for multiple meshes in preciceDict
    precicedict_str = r"""
    interfaces
    {
    Interface1
    {
        mesh              Fluid-Mesh-Centers;
        locations         faceCenters;
        connectivity      false;
        patches           (interface);

        // ... writeData, readData ...
    };

    Interface2
    {
        mesh              Fluid-Mesh-Nodes;
        locations         faceNodes;
        connectivity      true;
        patches           (interface);

        // ... writeData, readData ...
    };
    };
    """
    for idx_ in range(2):
        new_string = parallel_precice_dict(precicedict_str, idx_)
        print(new_string)

    precicedict_str = r"""
    interfaces
    {
    Interface1
    {
        mesh Fluid-Mesh;
        locations         faceCenters;
        patches           (cylinder_jet1);

        readData ( Velocity );

        writeData ( Pressure Temp);
    };

    Interface1
    {
        mesh Fluid-Meshh;
        locations         faceCenters;
        patches           (cylinder_jet1);

        readData ( Velocity Temp
        VarOn2ndLine Var2_Line2
        Var3
        );

        writeData ( Pressure_0 );
    };
    };
    """
    for idx_ in range(2):
        new_string = parallel_precice_dict(precicedict_str, idx_)
        print(new_string)

    output_filename = "precice-config_parallel_auto.xml"
    filename = "precice-config.xml"
    foldername = ""
    n_parallel_env = 4
    # loading the file at once might not be optimal for large files
    parallel_folders_list = [f'/data/ahmed/rl_play/examples/cylinder2D_openfoam/temp_{idx}' for idx in range(n_parallel_env)]
    parallel_tree = make_parallel_config(foldername, filename, n_parallel_env, parallel_folders_list, use_mapping=True)
    with open(output_filename, 'w') as output_file:
        output_file.write(xmltodict.unparse(parallel_tree, encoding='utf-8', pretty=True))
