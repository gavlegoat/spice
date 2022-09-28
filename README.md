# SPICE

SPICE is a tool for allowing reinforcement learning agents to explore their
environment safely, and is introduced in the paper "Guiding Safe Exploration
with Weakest Preconditions." This repository contains the implementation of
SPICE used for the experiments presented in that paper.

## Requirements

This code has been tested with Python 3.8.0. For these instructions we will
assume the command `python` refers to Python 3.8 and the command `pip` refers
to an appropriate version of pip. The required packages are listed in
`requirements.txt` and can be installed with

    pip install -r requirements.txt

SPICE also relies on the py-earth package for model learning. Code and
installation instructions can be found at the [py-earth
GitHub](https://github.com/scikit-learn-contrib/py-earth).

## Running

The entry point for all experiments is `main.py`. To replicate the experiments
from the paper, run

    python main.py --env_name acc --automatic_entropy_tuning

where `acc` may be replaced with the name of any other benchmark. To replicate
the conservative safety critic experiments, run

    python main.py --env_name acc --automatic_entropy_tuning --neural_safety

To see a full list of options, run `python main.py --help`.

## Acknowledgements

The code in `pytorch_soft_actor_critic` along with `main.py` is adapted from
<https://github.com/pranz24/pytorch-soft-actor-critic>. The implementation of
conservative safety critics (`--neural-safety`) is based on Bharadhwaj, et.
al., "Conservative safety critics for exploration," ICLR 2021.
