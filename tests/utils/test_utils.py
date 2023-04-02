
from os import chdir, path
import requests
import pytest


################################ TEST: FILEUTILS ################################

from gymprecice.utils.constants import MAX_ACCESS_WAIT_TIME
from gymprecice.utils.fileutils import (open_file, make_env_dir)
CONTENT = "content"

def test_open_file(tmpdir):
    test_dir = tmpdir.mkdir("test")
    test_file = test_dir.join("open_file.txt")
    test_file.write(CONTENT)
    file_obj = open_file(test_file)
    assert file_obj.readline() == CONTENT

def test_open_file_IOError(tmpdir):
    test_dir = tmpdir.mkdir("test")
    test_file_IOError = test_dir.join("open_file_IOError.txt")
    requests.get.side_effect = IOError
    with pytest.raises(IOError):
        open_file(test_file_IOError)

def test_make_env_dir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    chdir(test_dir)

    fluid_openfoam_dir = test_dir.mkdir("fluid-openfoam")
    fluid_openfoam_content = fluid_openfoam_dir.mkdir("content")
    fluid_openfoam_content_file = fluid_openfoam_content.join("info.txt")
    fluid_openfoam_content_file.write(CONTENT)

    solid_fenics_dir = test_dir.mkdir("solid-fenics")
    solid_fenics_content = solid_fenics_dir.mkdir("content")
    solid_fenics_content_file = solid_fenics_content.join("info.txt")
    solid_fenics_content_file.write(CONTENT)

    env_dir = test_dir.mkdir("env_0")
    solver_list = ["fluid-openfoam", "solid-fenics"]
    make_env_dir(env_dir, solver_list)

    assert path.islink("env_0/fluid-openfoam/content/info.txt") is True 
    assert path.islink("env_0/solid-fenics/content/info.txt") is True
    
def test_make_env_dir_IOError(tmpdir):
    test_dir = tmpdir.mkdir("test")
    chdir(test_dir)

    fluid_openfoam_dir = test_dir.mkdir("fluid-openfoam")
    fluid_openfoam_content = fluid_openfoam_dir.mkdir("content")
    fluid_openfoam_content_file = fluid_openfoam_content.join("info.txt")
    fluid_openfoam_content_file.write(CONTENT)

    env_dir = test_dir.mkdir("env_0")
    solver_list = ["fluid-openfoam", "solid-fenics"]
    
    requests.get.side_effect = Exception
    with pytest.raises(Exception):
        make_env_dir(env_dir, solver_list)


################################ TEST: XMLUTILS ################################

from gymprecice.utils.xmlutils import (_load_file, get_episode_end_time, get_mesh_data, set_training_dir)

@pytest.mark.skip(reason="to be filled shell")
def test_load_file():
    pass

@pytest.mark.skip(reason="to be filled shell")
def test_get_episode_end_time():
    pass

@pytest.mark.skip(reason="to be filled shell")
def test_get_mesh_data():
    pass

@pytest.mark.skip(reason="to be filled shell")
def test_set_training_dir():
    pass

