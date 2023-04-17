import re
from setuptools import setup, find_packages
import sys


# adapted from https://stackoverflow.com/a/9079062
if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] != 8):
    raise Exception("Error: gymprecice only supports Python 3.8.")

with open("gymprecice/version.py") as file:
    full_version = file.read()
    assert (
        re.match(r'VERSION = "\d\.\d+\.\d+"\n', full_version).group(0) == full_version
    ), f"Unexpected version: {full_version}"
    VERSION = re.search(r"\d\.\d+\.\d+", full_version).group(0)

# Uses the readme as the description on PyPI
with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

# All dependencies
_deps = [
    # core
    "gymnasium==0.28.0",
    "pyprecice==2.4.0.0",
    "xmltodict>=0.13.0",
    "psutil>=5.9.2",
    # testing 
    "pytest==7.1.3",
    "pytest-mock==3.10.0",
    # tutorial
    "matplotlib>=3.0",
    "torch==1.12.1",
    "scipy>=1.7.3",
    # style
    "black~=23.1",
    # training
    "wandb>=0.13.6",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

extras = {}
extras["testing"] = deps_list("pytest", "pytest-mock")
extras["tutorial"] = deps_list("torch", "matplotlib", "scipy")
extras["style"] = deps_list("black")
extras["training"] = deps_list("wandb")


install_requires = [
    deps["gymnasium"],
    deps["pyprecice"],
    deps["xmltodict"],
    deps["psutil"],
]

setup(
    name="gymprecice",
    version=VERSION,
    license="MIT",
    url="https://github.com/gymprecice/gymprecice/",
    description="Gym-preCICE is a preCICE adapter that provides a Gymnasium-like API to couple reinforcement learning and physics-based solvers for active control",
    author="Mosayeb Shams (lead-developer), Ahmed. H. Elsheikh (co-developer and supervisor)",
    author_email="m.shams@hw.ac.uk, a.elsheikh@hw.ac.uk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires="==3.8.*",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras,
    test_suite="tests",
    classifiers = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",     
    ],
)
