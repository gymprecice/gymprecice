import itertools
import re

from setuptools import setup, find_packages


# adapted from https://stackoverflow.com/a/9079062
import sys
if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 7):
    raise Exception("Error: gymprecice only supports Python 3.7 and higher.")

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

# # Specific dependencies.
extras = {}

# Visualisation dependency groups.
testing_group = set(extras.keys())
extras["visual"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], testing_group)))
) + ["wandb>=0.13.6"]

# Testing dependency groups.
testing_group = set(extras.keys())
extras["testing"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], testing_group)))
) + ["pytest==7.0.1", "mock==5.0.1"]

setup(
    name='gymprecice',
    version=VERSION,
    license="MIT",
    author='Mosayeb Shams (lead-developer), Ahmed. H. Elsheikh (co-developer and supervisor)',
    author_email='m.shams@hw.ac.uk, a.elsheikh@hw.ac.uk',
    description='Gym-preCICE is a preCICE adapter that provides a Gymnasium-like API to couple reinforcement learning and physics-based solvers for active control',
    long_description=long_description,
    long_description_content_type='text/markdown',
    extras_require=extras,
    tests_require=extras["testing"],
    install_requires=[
        "gymnasium>=0.26.0",
        "torch==1.12.1",
        "pyprecice==2.4.0.0",
        "scipy>=1.7.3",
        "xmltodict>=0.13.0",
        "psutil>=5.9.2",
    ],
    packages=find_packages(),
    package_data={
        "gymprecice": [
            "envs/openfoam/**/*",
        ]
    },
    url="https://github.com/gymprecice/gymprecice/",
    zip_safe=False,
)