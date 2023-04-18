[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/gymprecice/gymprecice/blob/master/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Run Pre-Commit: passing](https://github.com/gymprecice/gymprecice/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/gymprecice/gymprecice/actions/workflows/pre-commit.yml)
## Gym-preCICE

Gym-preCICE is a Python [preCICE](https://github.com/precice/precice) adapter fully compliant with the 
[Gymnasium API](https://github.com/Farama-Foundation/Gymnasium), aka [OpenAI Gym](https://github.com/openai/gym), 
to facilitate employing and developing reinforcement learning algorithms for single- and multi-physics active control applications. 
In a Reinforcement Learning-interaction cycle, Gym-preCICE takes advantage of coupling tool preCICE, an open-source library for multi-physics coupling, to handle information exchange between a Reinforcement Learning agent ("controller") and external mesh-based solvers ("physics-simulation-engine"). The primary use of Gym-preCICE adapter is for closed- and open-loop active control of physics simulations.

## Installation

### Main required dependencies
**Gymnasium**:  Installed by default. Refer to [the Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for more information.

**preCICE**: You need to install the preCICE library. Refer to [the preCICE documentation](https://precice.org/installation-overview.html) for information on building and installation.

**preCICE Python bindings**: Installed by default. Refer to [the python language bindings for preCICE](https://github.com/precice/python-bindings) for information.

### Installing the package
We support and test for Python 3.8 on Linux. We recommend installing Gym-preCICE within a virtual environment using [conda](https://www.anaconda.com/products/distribution#Downloads) with `Python 3.8`:

- create and activate a conda virtual environment:
```bash
 conda create -n gymprecice python=3.8
 conda activate gymprecice
```
- install the adapter:
```bash
python3 -m pip install gymprecice
```
- run a simple test to check `gymprecice` installation (this should pass silently without any error/warning messages):
```bash
python3 -c "import gymprecice"
```
The above installation does not include extra dependencies to run tests or tutorials. You can install these dependencies like `python3 -m pip install gymprecice[testing]`, or 
`python3 -m pip install gymprecice[tutorial]`, or use `python3 -m pip install gymprecice[all]` to install all extra dependencies. 

### Testing
We use `pytest` testing framework to write and execute unit tests for all modules in our package. You need to install 
required dependencies before running any test:
```bash
python3 -m pip install gymprecice[testing]
``` 
- run the full test suits:
```
pytest ./tests
```
It is also possible to test individual modules by providing the path to the respective test script to `pytest`. For instance, to test the core module (`core.py`):
```
pytest ./tests/test_core.py
```
### Usage
Please refer to [gymprecice-tutorials](https://github.com/gymprecice/gymprecice-tutorials) for the details on how to use the adapter. You can check out the [Quickstart](https://github.com/gymprecice/gymprecice-tutorials/tree/main/quickstart) in our [gymprecice-tutorials](https://github.com/gymprecice/gymprecice-tutorials) repository to launch a control case. You need to install 
required dependencies before running any tutorial:
```bash
python3 -m pip install gymprecice[tutorial]
``` 

## Contributions

gymprecice is currently developed and maintained by: 

- Mosayeb Shams (@mosayebshams) - Lead developer
- Ahmed H. Elsheikh(@ahmed-h-elsheikh) - Co developer and Supervisor 

## Citation
If you use Gym-preCICE, please consider citing its technical paper:

```
@misc{,
  Author = {Mosayeb Shams and Ahmed H. Elsheikh},
  Title = {Gym-preCICE: Coupling Reinforcement Learning Algorithms with External Physics-Based Solvers for Active Flow Control},
  Year = {2023},
  Eprint = {arXiv:},
}
```

## License

gymprecice is MIT-licensed; Please refer to the [LICENSE](https://github.com/gymprecice/blob/main/LICENSE) file for more information.
