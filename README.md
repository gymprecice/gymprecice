## Gym-preCICE

Gym-preCICE is a Python [preCICE](https://github.com/precice/precice) adapter fully compliant with
the [Gymnasium API](https://github.com/Farama-Foundation/Gymnasium) to facilitate employing and developing reinforcement
learning algorithms for single- and multi-physics active flow control appli-
cations. In a reinforcement learning-interaction cycle, Gym-preCICE takes
advantage of coupling tool preCICE, an open-source library for multi-physics
coupling, to handle information exchange between agent (decision maker) and
external mesh-based solvers (simulation environment) within reinforcement
learning interaction cycles. 

## Installation

### Python environment

It is recommended to install the package within a virtual Python environment using [conda](https://docs.conda.io/en/latest/miniconda.html):
```
...
```

### Testing

To run all or selected tests:
```
...
```

### Training Example

Gym-preCICE includes the following families of environments:
* [OpenFOAM](https://github.com/gymprecice/gymprecice/envs/openfoam/) - OpenFOAM-based environments are only tested against **OpenFOAM-v2112**;
* [FSI](https://github.com/gymprecice/gymprecice/envs/fsi/) - 


Example:
```
...
```

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

gymprecice is [MIT](https://en.wikipedia.org/wiki/MIT_License)-licensed; refer to the [LICENSE](https://github.com/gymprecice/blob/main/LICENSE) file for more information.