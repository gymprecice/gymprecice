[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/gymprecice/gymprecice/blob/master/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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
- clone the repository, switch to the root directory, and install the adapter:
```bash
git clone --single-branch --branch main https://github.com/gymprecice/gymprecice
cd gymprecice
pip install .
```
- run a simple test to check `gymprecice` installation (this should pass silently without any error/warning messages):
```bash
python3 -c "import gymprecice"
```
The above installation does not include extra dependencies such as `torch`, `pytest`, etc. You can install these dependencies like `pip install .[torch]` or use `pip install .[all]` to install all extra dependencies. 

### Testing
We use `pytest` testing framework to write and execute unit tests for all modules in our package.  
- switch to the root directory (`gymprecice`) and install testing dependencies:
```bash
pip install .[testing]
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
Users of the adapter need to define their control problem in a simple structure consisting of three key components ([please see gymprecice-tutorials](https://github.com/gymprecice/gymprecice-tutorials/tree/master/closed_loop_AFC/rotating_control_cylinder)):
- `physics-simulation-engine`: a directory containing PDE-based solver case(s), `gymprecice-config.json` file for the adapter configuration, and `precice-config.xml` file for configuring the preCICE coupling library.
- `environment.py`: a Python script defining a class inherited from Gym-preCICE adapter to expose the underlying behaviour of the physics-simulation-engine to the controller.
- `controller.py`: a Python script defining the controller algorithm that interacts with the environment. This may, for instance, be the Proximal Policy Optimisation (PPO) algorithm, the Soft-Actor-Critic (SAC) algorithm, or a simple sinusoidal signal control.

```bash
control-problem
├── controller.py
├── envrionment.py
└── physics-simulation-engine
    ├── gymprecice-config.json
    ├── precice-config.json
    ├── solver-1
    ├── solver-2
    └── solver-n
```
To run the control case, you need to switch to the root directory of the control case, here, `control-problem`, and run
 ```bash
 python3 -u controller.py
 ```
By default, the output will be saved in a directory called `gymprecice-run` that is located in the root directory of the control case. However, it is possible to specify a different path for the result directory via `gymprecice-config.json` file.

Check out the [Quickstart](https://github.com/gymprecice/gymprecice-tutorials) in our gymprecice-tutorials repository to launch a control case.

## Contributions

gymprecice is currently developed and maintained by: 

- Mosayeb Shams (@mosayebshams) - Lead developer
- Ahmed H. Elsheikh(@ahmed-h-elsheikh) - Co developer and Supervisor 

## Citation
If you use Gym-preCICE, please cite its technical paper with the following bibtex entry:

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
