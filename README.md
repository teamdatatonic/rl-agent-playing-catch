# Reinforcement Learning Catch Example
This is the repository for the reinforment learning catch example. In this project we used RL to train an agent controlling a basket to catch a piece of falling fruit. We define an environment entirely using matplotlib. 

Click [here](https://docs.google.com/document/d/1xg5XOEiPGzym0GEzzZr-oLxU_tabhzpNmsGbag_-YIE/edit#heading=h.kl0urft1gbwy) to read more about the theory behind the RL agent and the learning process.
# Getting Started
## Pre-requisites
* Python 3.9.0
* [pyenv](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)
* poetry

For Unix users, we recommend the use of `pyenv` to manage the Python version as specifed in `.python-version`. See the [installation instruction](https://github.com/pyenv/pyenv#installation) for setting up `pyenv` on your system.

* Install poetry `pip install poetry`
* Install pyenv `pip install pyenv`
## Installation
* For first time users in the root directory run:
    1. `poetry install` to install all depenencies
    2. `pyenv install` to install and use the correct version of python

## Running the code 
* To run the train script locally run `poetry run python src/train.py`
* To run the inference script locally:
    1. Make sure you have a model file stored as `model/model.h5`
    2. Run `poetry run python src/run.py`

## Docker (TODO)
