
import pytest
import requests

from os import chdir, path, listdir

from gymprecice.utils.constants import MAX_ACCESS_WAIT_TIME
from gymprecice.utils.fileutils import (open_file, make_env_dir, make_result_dir)

FILE_CONTENT = "content"

def test_vaild_open_file(tmpdir):
    test_dir = tmpdir.mkdir("test")
    valid_input = test_dir.join("open_file.txt")
    valid_input.write(FILE_CONTENT)
    output = open_file(valid_input)
    assert output.readline() == FILE_CONTENT

def test_invalid_open_file(tmpdir):
    test_dir = tmpdir.mkdir("test")
    invalid_input = test_dir.join("open_file_IOError.txt")  # file does not exist
    requests.get.side_effect = IOError
    with pytest.raises(IOError):
        open_file(invalid_input)

def test_valid_make_env_dir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    chdir(test_dir)

    fluid_openfoam_dir = test_dir.mkdir("fluid-openfoam")
    fluid_openfoam_content = fluid_openfoam_dir.mkdir("content")
    fluid_openfoam_content_file = fluid_openfoam_content.join("info.txt")
    fluid_openfoam_content_file.write(FILE_CONTENT)

    solid_fenics_dir = test_dir.mkdir("solid-fenics")
    solid_fenics_content = solid_fenics_dir.mkdir("content")
    solid_fenics_content_file = solid_fenics_content.join("info.txt")
    solid_fenics_content_file.write(FILE_CONTENT)

    valid_input_0 = test_dir.mkdir("env_0")
    valid_input_1 = ["fluid-openfoam", "solid-fenics"]
    make_env_dir(valid_input_0 , valid_input_1)

    output = {
        "soft_link_1_bool": path.islink("env_0/fluid-openfoam/content/info.txt"),
        "soft_link_2_bool": path.islink("env_0/solid-fenics/content/info.txt")
    }
    expect = {
        "soft_link_1_bool": True,
        "soft_link_2_bool": True
    }
    assert output == expect
    
def test_invalid_make_env_dir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    chdir(test_dir)

    fluid_openfoam_dir = test_dir.mkdir("fluid-openfoam")
    fluid_openfoam_content = fluid_openfoam_dir.mkdir("content")
    fluid_openfoam_content_file = fluid_openfoam_content.join("info.txt")
    fluid_openfoam_content_file.write(FILE_CONTENT)

    valid_input = test_dir.mkdir("env_0")
    invalid_input = ["fluid-openfoam", "invalid-solver"]
    
    requests.get.side_effect = Exception
    with pytest.raises(Exception):
        make_env_dir(valid_input, invalid_input)

def test_valid_make_result_dir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    chdir(test_dir)

    test_env_dir = test_dir.mkdir("test_env_dir")
    valid_input = {
        "environment":
        {
            "name": "test_env"
        },
        "solvers":
        {
            "name": ["fluid-openfoam", "solid-fenics"],
        },
        "precice":
        {
            "precice_config_file_name": "precice-config.xml"
        }
    }
    valid_input["environment"]["src"] = test_env_dir

    xml_content = \
    '''<?xml version="1.0"?>
    ...
        <m2n:sockets from="Controller" to="Fluid" exchange-directory=""/>
        <m2n:sockets from="Controller" to="Solid" exchange-directory=""/>
    ...'''

    precice_config_file =  test_env_dir.join("precice-config.xml")
    precice_config_file.write(xml_content)

    fluid_openfoam_dir = test_env_dir.mkdir("fluid-openfoam")
    fluid_openfoam_content = fluid_openfoam_dir.mkdir("content")
    fluid_openfoam_content_file = fluid_openfoam_content.join("info.txt")
    fluid_openfoam_content_file.write(FILE_CONTENT)

    solid_fenics_dir = test_env_dir.mkdir("solid-fenics")
    solid_fenics_content = solid_fenics_dir.mkdir("content")
    solid_fenics_content_file = solid_fenics_content.join("info.txt")
    solid_fenics_content_file.write(FILE_CONTENT)

    make_result_dir(valid_input)

    chdir(test_dir)
    run_dir = path.join("gymprecice-run", listdir("gymprecice-run")[0])
    
    output = {
        "gymprecice-run": path.exists("gymprecice-run"),
        "precice-config.xml": path.exists(path.join(run_dir, "precice-config.xml")),
        "fluid-openfoam": path.exists(path.join(run_dir, "fluid-openfoam")),
        "solid-fenics": path.exists(path.join(run_dir, "solid-fenics")),
        "fluid-openfoam-content": path.exists(path.join(run_dir, "fluid-openfoam", "content", "info.txt")),
        "solid-fenics-content": path.exists(path.join(run_dir, "solid-fenics", "content", "info.txt")),
    }
    expect = {
        "gymprecice-run": True,
        "precice-config.xml": True,
        "fluid-openfoam": True,
        "solid-fenics": True,
        "fluid-openfoam-content": True,
        "solid-fenics-content": True
    }
    assert output == expect
