import os
from setuptools import setup, find_packages
import warnings

# TODO: execlude __pycache__

# from https://stackoverflow.com/a/9079062
import sys
if sys.version_info[0] < 3:
    raise Exception("gymprecice only supports Python3.9. Did you run $python setup.py <option>.? "
                    "Try running $python3 setup.py <option>.")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gymprecice',
    version='v0.0.1',
    description='gymprecice is an OpenAI like environment for control of couple problems using preCICE',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gymprecice',
    entry_points={
        'console_scripts': ['gymprecice=gymprecice.gymprecice:main']},
    author='Ahmed H. Elsheikh, Mosayeb Shams',
    author_email='a.elsheikh@hw.ac.uk, m.shams@hw.ac.uk',
    license='LGPL-3.0',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'pyprecice>=2.4.0',
        'numpy>=1.13.3',
        'mpi4py'],
    test_suite='tests',
    zip_safe=False)
