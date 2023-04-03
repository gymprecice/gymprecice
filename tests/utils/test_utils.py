
from os import chdir, path, listdir
import requests
import pytest
from sys import stderr
import copy


################################ TEST: FILEUTILS ################################

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

    
################################ TEST: XMLUTILS ################################

from gymprecice.utils.xmlutils import (get_episode_end_time, get_mesh_data)
VALID_XML_CONTENT_0 = \
'''<?xml version="1.0"?>
<precice-configuration>
    ...
    <solver-interface dimensions="2">
        <data:vector name="Velocity" />
        <data:scalar name="Pressure" />
        ...
        <mesh name="Fluid-Mesh">
            <use-data name="Velocity" />
            <use-data name="Displacement" />
        </mesh>
        <mesh name="Controller-Mesh">
            <use-data name="Velocity" />
            <use-data name="Pressure" />
        </mesh>
        <participant name="Fluid">
        ...
        </participant>
        <participant name="Controller">
            <use-mesh name="Controller-Mesh" provide="yes" />
            <use-mesh name="Fluid-Mesh" from="Fluid"/>
            <write-data name="Velocity" mesh="Controller-Mesh" />
            <read-data name="Pressure"  mesh="Fluid-Mesh" />
        </participant>
        ...
        <coupling-scheme:parallel-explicit>
            <max-time value="2.335" />
            <time-window-size value="0.0005" valid-digits="8" />
            ...
        </coupling-scheme:parallel-explicit>
            ...
    </solver-interface>
</precice-configuration>'''

EXPECTED_0 = {
    "scaler_list": ['Pressure'],
    "vector_list": ['Velocity'],
    "mesh_lis": ['Fluid-Mesh', 'Controller-Mesh'],
    "controller_dict": {
        "Fluid-Mesh": {"read": ["Pressure"], "write": []}, 
        "Controller-Mesh": {"read": [], "write": ["Velocity"]}
    }
}

VALID_XML_CONTENT_1 = \
'''<?xml version="1.0"?>
<precice-configuration>
    ...
    <solver-interface dimensions="2">
        <data:vector name="Velocity" />
        <data:scalar name="Pressure" />
        <data:scalar name="Displacement" />
        <data:vector name="Force" />
        ...
        <mesh name="Fluid-Mesh">
            <use-data name="Velocity" />
            <use-data name="Displacement" />
        </mesh>
        <mesh name="Solid-Mesh">
            <use-data name="Force" />
        </mesh>
        <mesh name="Controller-Mesh">
            <use-data name="Velocity" />
            <use-data name="Pressure" />
            <use-data name="Displacement" />
        </mesh>
        ...
        <participant name="Fluid">
        ...
        </participant>
        <participant name="Solid">
        ...
        </participant>
        <participant name="Controller">
            <use-mesh name="Controller-Mesh" provide="yes" />
            <use-mesh name="Fluid-Mesh" from="Fluid"/>
            <use-mesh name="Solid-Mesh" from="Solid"/>
            <write-data name="Velocity" mesh="Controller-Mesh" />
            <read-data name="Pressure"  mesh="Fluid-Mesh" />
            <write-data name="Force" mesh="Controller-Mesh" />
            <read-data name="Displacement" mesh="Solid-Mesh" />
        </participant>
        ...
    </solver-interface>
</precice-configuration>'''

EXPECTED_1 = {
    "scaler_list": ['Pressure', 'Displacement'],
    "vector_list": ['Velocity', 'Force'],
    "mesh_lis": ['Fluid-Mesh', 'Solid-Mesh', 'Controller-Mesh'],
    "controller_dict": {
        "Fluid-Mesh": {"read": ["Pressure"], "write": []}, 
        "Solid-Mesh": {"read": ["Displacement"], "write": []},
        "Controller-Mesh": {"read": [], "write": ["Velocity", "Force"]}
    }
}

def test_valid_get_episode_end_time(tmpdir):
    test_dir = tmpdir.mkdir("test")
    valid_input = test_dir.join("precice-config.xml")
    valid_input.write(VALID_XML_CONTENT_0)
    episode_end_time = get_episode_end_time(valid_input)
    assert episode_end_time == 2.335

@pytest.mark.parametrize(
    "test_input, expected",
    [(VALID_XML_CONTENT_0, EXPECTED_0), (VALID_XML_CONTENT_1, EXPECTED_1)],
)
def test_valid_get_mesh_data(tmpdir, test_input, expected):
    test_dir = tmpdir.mkdir("test")
    valid_input = test_dir.join("precice-config.xml")
    valid_input.write(test_input)

    scaler_list, vector_list, mesh_list, controller_dict = get_mesh_data(valid_input)
    output = {
        "scaler_list": scaler_list,
        "vector_list": vector_list,
        "mesh_lis": mesh_list,
        "controller_dict": controller_dict
    }
    assert output == expected

def test_invalid_get_mesh_data(tmpdir):
    test_dir = tmpdir.mkdir("test")
    invalid_input = test_dir.join("precice-config.xml")

    invalid_xml_content = \
    '''<?xml version="1.0"?>
    <precice-configuration>
        ...
        <solver-interface dimensions="2">
            ...
            <mesh name="Controller-Mesh">
                ...
            </mesh>
            ...
            <participant name="Controller">
                ...
            </participant>
            ...
        </solver-interface>
    </precice-configuration>'''
    invalid_input.write(invalid_xml_content)

    requests.get.side_effect = AssertionError
    with pytest.raises(AssertionError):
        get_mesh_data(invalid_input)

