from gymprecice.utils.constants import FILE_ACCESS_SLEEP_TIME
import re
from time import sleep
import os
import numpy as np

verbose_mode = False


import struct
from collections import namedtuple
Boundary = namedtuple('Boundary', 'type, num, start, id')


def _is_integer(s):
    try:
        x = int(s)
        return True
    except ValueError:
        return False


def _is_binary_format(content, maxline=20):
    """
    parse file header to judge the format is binary or not
    :param content: file content in line list
    :param maxline: maximum lines to parse
    :return: binary format or not
    """
    for lc in content[:maxline]:
        if b'format' in lc:
            if b'binary' in lc:
                return True
            return False
    return False


def _parse_boundary_content(content, is_binary=None, skip=10):
    """
    parse boundary from content
    :param content: file contents
    :param is_binary: binary format or not, not used
    :param skip: skip lines
    :return: boundary dict
    """
    bd = {}
    num_boundary = 0
    n = skip
    bid = 0
    in_boundary_field = False
    in_patch_field = False
    current_patch = b''
    current_type = b''
    current_nFaces = 0
    current_start = 0
    while True:
        if n > len(content):
            if in_boundary_field:
                print('error, boundaryField not end with )')
            break
        lc = content[n]
        if not in_boundary_field:
            if _is_integer(lc.strip()):
                num_boundary = int(lc.strip())
                in_boundary_field = True
                if content[n + 1].startswith(b'('):
                    n += 2
                    continue
                elif content[n + 1].strip() == b'' and content[n + 2].startswith(b'('):
                    n += 3
                    continue
                else:
                    print('no ( after boundary number')
                    break
        if in_boundary_field:
            if lc.startswith(b')'):
                break
            if in_patch_field:
                if lc.strip() == b'}':
                    in_patch_field = False
                    bd[current_patch] = Boundary(current_type, current_nFaces, current_start, -10-bid)
                    bid += 1
                    current_patch = b''
                elif b'nFaces' in lc:
                    current_nFaces = int(lc.split()[1][:-1])
                elif b'startFace' in lc:
                    current_start = int(lc.split()[1][:-1])
                elif b'type' in lc:
                    current_type = lc.split()[1][:-1]
            else:
                if lc.strip() == b'':
                    n += 1
                    continue
                current_patch = lc.strip()
                if content[n + 1].strip() == b'{':
                    n += 2
                elif content[n + 1].strip() == b'' and content[n + 2].strip() == b'{':
                    n += 3
                else:
                    print('no { after boundary patch')
                    break
                in_patch_field = True
                continue
        n += 1

    return bd


def _parse_faces_content(content, is_binary, skip=10):
    """
    parse faces from content
    :param content: file contents
    :param is_binary: binary format or not
    :param skip: skip lines
    :return: faces as list
    """
    n = skip
    while n < len(content):
        lc = content[n]
        if _is_integer(lc):
            num = int(lc)
            if not is_binary:
                data = [[int(s) for s in ln[2:-2].split()] for ln in content[n + 2:n + 2 + num]]
            else:
                buf = b''.join(content[n+1:])
                disp = struct.calcsize('c')
                idx = struct.unpack('{}i'.format(num), buf[disp:num*struct.calcsize('i') + disp])
                disp = 3*struct.calcsize('c') + 2*struct.calcsize('i')
                pp = struct.unpack('{}i'.format(idx[-1]),
                                    buf[disp+num*struct.calcsize('i'):
                                        disp+(num+idx[-1])*struct.calcsize('i')])
                data = []
                for i in range(num - 1):
                    data.append(pp[idx[i]:idx[i+1]])
            return data
        n += 1
    return None


def _parse_points_content(content, is_binary, skip=10):
    """
    parse points from content
    :param content: file contents
    :param is_binary: binary format or not
    :param skip: skip lines
    :return: points coordinates as numpy.array
    """
    n = skip
    while n < len(content):
        lc = content[n]
        if _is_integer(lc):
            num = int(lc)
            if not is_binary:
                data = np.array([ln[1:-2].split() for ln in content[n + 2:n + 2 + num]], dtype=float)
            else:
                buf = b''.join(content[n+1:])
                disp = struct.calcsize('c')
                vv = np.array(struct.unpack('{}d'.format(num*3),
                                            buf[disp:num*3*struct.calcsize('d') + disp]))
                data = vv.reshape((num, 3))
            return data
        n += 1
    return None


def _parse_mesh_file(fn, parser):
    """
    parse mesh file
    :param fn: boundary file name
    :param parser: parser of the mesh
    :return: mesh data
    """
    try:
        with open(fn, "rb") as f:
            content = f.readlines()
            return parser(content, _is_binary_format(content))
    except FileNotFoundError:
        print('file not found: %s' % fn)
        return None


def _parse_mesh_data(path):
    """
    parse mesh data from mesh files
    :param path: path of mesh files
    :return: none
    """
    boundary = _parse_mesh_file(os.path.join(path, 'boundary'), _parse_boundary_content)
    points = _parse_mesh_file(os.path.join(path, 'points'), _parse_points_content)
    faces = _parse_mesh_file(os.path.join(path, 'faces'), _parse_faces_content)

    return boundary, points, faces


def _boundary_face_centre(path, patch):
    """
    return coordinate of faces on boundary patch
    :param patch: boundary name, byte str
    """
    boundary, points, faces = _parse_mesh_data(path)
    
    try:
        b = boundary[patch]
        faces = np.array([faces[f] for f in range(b.start, b.start+b.num)])
        face_centres = []

        for face in faces:
            vertices = np.array([points[idx] for idx in face])
            n_vertices = face.size
            if n_vertices == 3:
                face_centres.append(vertices.sum(axis=0) / 3)
            else:
                center_point = vertices.sum(axis=0) / n_vertices
                vertex_idx = 0
                sum_area = 0
                sum_area_centre = 0
                for vertex in vertices:
                    next_vertex = vertices[(vertex_idx + 1) % n_vertices]
                    triangle_centre = vertex + next_vertex + center_point
                    normal_vector = np.cross(vertex-center_point, next_vertex-center_point)
                    area = np.sqrt(normal_vector.dot(normal_vector))

                    sum_area += area
                    sum_area_centre += area * triangle_centre
                    vertex_idx += 1

                if sum_area > 1e-16:
                    face_centres.append(sum_area_centre / (3*sum_area))
                else:
                    return face_centres.append(center_point)
        return np.array(face_centres)

    except KeyError:
        return ()


def _boundary_face_area(path, patch):
    """
    return coordinate of faces on boundary patch
    :param patch: boundary name, byte str
    """
    boundary, points, faces = _parse_mesh_data(path)
    
    try:
        b = boundary[patch]
        faces = np.array([faces[f] for f in range(b.start, b.start+b.num)])
        face_areas = []
        face_vector_areas = []
        face_normals = []

        total_area = 0
        for face in faces:

            vertices = np.array([points[idx] for idx in face])
            n_vertices = face.size
            if n_vertices == 3:
                normal_vector = np.cross(vertices[0]-vertices[2], vertices[1]-vertices[2])
                n = normal_vector/np.sqrt(normal_vector.dot(normal_vector))
                face_normals.append(n)
                area = 0.5 * np.sqrt(normal_vector.dot(normal_vector))
                face_areas.append(area)
                face_vector_areas.append(area * n)
                total_area += area

            else:
                center_point = vertices.sum(axis=0) / n_vertices
                vertex_idx = 0
                sum_area = 0
                sum_normals = np.zeros(3)
                for vertex in vertices:
                    next_vertex = vertices[(vertex_idx + 1) % n_vertices]
                    normal_vector = np.cross(vertex-center_point, next_vertex-center_point)
                    sum_normals += normal_vector
                    area = 0.5*np.sqrt(normal_vector.dot(normal_vector))
                    sum_area += area
                    vertex_idx += 1

                total_area += sum_area
                face_areas.append(sum_area)
                n = sum_normals/np.sqrt(sum_normals.dot(sum_normals))
                face_normals.append(n)
                face_vector_areas.append(sum_area * n)

        # print(f"total area of patch: {total_area}")
        return np.array(face_vector_areas), np.array(face_areas), np.array(face_normals)

    except KeyError:
        return ()


def get_patch_geometry(case_path, patches):
    path = os.path.join(case_path, "constant/polyMesh/")
    patch_data = {}
    for patch in patches:
        Cf = _boundary_face_centre(path, patch.encode())
        Sf, magSf, nf = _boundary_face_area(path, patch.encode())
        patch_data[patch] = {'face_centre': Cf, 'face_area_vector': Sf, 'face_area_mag': magSf, 'face_normal': nf}
    return patch_data


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


def robust_readline(filehandler, n_expected):
    file_pos = filehandler.tell()
    line_text = filehandler.readline()
    is_comment, time_idx, n_probes, probe_data = parse_probe_lines(line_text.strip())
    if not is_comment and n_probes != n_expected:
        filehandler.seek(file_pos)
        sleep(FILE_ACCESS_SLEEP_TIME)
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
