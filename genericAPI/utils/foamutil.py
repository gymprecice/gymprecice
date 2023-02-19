import re
def _get_actuation_patches(path_to_precicedict):
    """ 
    Extract patch names of the actuation boundaries.  

    Parameters
    ----------
    path_to_preciceDict : path to the precideDict file in OpenFOAM case.

    Returns
    -------
    name_list : str list
        List of names of actuation patches.

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

from utils.mesh_parser import FoamMesh  # TODO: implement this as a stand alone method
def _get_patch_geometry(case_path, patches):
    foam_mesh = FoamMesh(case_path)
    patch_data = {}
    for patch in patches:
        Cf = foam_mesh.boundary_face_centres(patch.encode())        
        Sf, magSf, nf = foam_mesh.boundary_face_area(patch.encode())
        patch_data[patch] = {'Cf': Cf, 'Sf': Sf, 'magSf': magSf, 'nf': nf}
    return patch_data

import os
from time import sleep
def _make_env_dir(env_dir, case_dir):
    os.system(f'rm -rf {os.path.join(os.getcwd(), env_dir)}')
    try:
        os.system(f'cp -rs {os.path.join(os.getcwd(), case_dir)} {env_dir}')
    except Exception as e:
        raise Exception(f'Failed to create symbolic links to foam case files: {e}')
    sleep(0.5)

