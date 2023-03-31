from gymprecice.envs.openfoam.mesh_parser import FoamMesh  # TODO: implement this as a stand alone method
import re
import time

verbose_mode = False


def get_interface_patches(path_to_precicedict):
    """
    Extract patch names of the interface boundaries.

    Parameters
    ----------
    path_to_preciceDict : path to the precideDict file in OpenFOAM case.

    Returns
    -------
    name_list : str list
        List of names of interface patches.

    """
    # read the file content as a string
    precicedict_str = None
    with open(path_to_precicedict, 'r') as filehandle:
        precicedict_str = filehandle.readlines()
    precicedict_str = '\n'.join(precicedict_str)

    # find acuator patch names
    splitted_list = re.split(r"patches\s*\(\s*(.*?)\s*\);", precicedict_str)
    name_list = []
    for matched_str in splitted_list[1::2]:
        local_list = [patch_name for patch_name in re.split(r"\s+", matched_str)]
        name_list += local_list
    return name_list


def get_patch_geometry(case_path, patches):
    foam_mesh = FoamMesh(case_path)
    patch_data = {}
    for patch in patches:
        Cf = foam_mesh.boundary_face_centres(patch.encode())
        Sf, magSf, nf = foam_mesh.boundary_face_area(patch.encode())
        patch_data[patch] = {'Cf': Cf, 'Sf': Sf, 'magSf': magSf, 'nf': nf}
    return patch_data


def robust_readline(filehandler, n_expected, sleep_time=0.01):
    file_pos = filehandler.tell()
    line_text = filehandler.readline()
    is_comment, time_idx, n_probes, probe_data = parse_probe_lines(line_text.strip())
    if not is_comment and n_probes != n_expected:
        # print(f'Reading a line expected number of fields vs read: {n_expected}, {n_probes} -- wait for a bit')
        # print(line_text)
        filehandler.seek(file_pos)
        time.sleep(sleep_time)
    return is_comment, time_idx, n_probes, probe_data


def parse_probe_lines(line_string):
    if len(line_string) == 0:
        # print('line of length zero')
        return False, None, 0, None
    if line_string[0] == "#":
        if verbose_mode:
            print(f"comment line: {line_string}")
        is_comment = True
        return is_comment, None, 0, None

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
        if verbose_mode:
            print("vector variable")
        num_probes = line_string.count("(")
        assert num_probes == line_string.count(")"), f'corrupt file, number of ( and ) should be equal:" \
            "{line_string.count(")")}, {line_string.count(")")}'
        assert (len(float_list) - 1) % num_probes == 0, f'corrupt file, each probe should have the same number of components, {len(float_list)}, {num_probes}'
    else:
        num_probes = len(float_list) - 1
    # comment or not, time idx, number of probes, probe values
    return is_comment, float_list[0], num_probes, float_list[1:]
