# Reinforcement Learning Catch Example
This is the repository for the reinforment learning catch example. In this project we used RL to train an agent controlling a basket to catch a piece of falling fruit. We define an environment entirely using matplotlib. 

Click [here](https://docs.google.com/document/d/1xg5XOEiPGzym0GEzzZr-oLxU_tabhzpNmsGbag_-YIE/edit#heading=h.kl0urft1gbwy) to read more about the theory behind the RL agent and the learning process.
# Getting Started
## Pre-requisites
* Python 3.9.6
* [pyenv](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)
* poetry

For Unix users, we recommend the use of `pyenv` to manage the Python version as specifed in `.python-version`. See below for instructions on setting up `poetry` and `pyenv` on your system.

### For MacOS:
* Install poetry `pip install poetry`
* Install [pyenv](https://github.com/pyenv/pyenv#homebrew-in-macos)

### For Windows:
* Install poetry `pip install poetry`
* Pyenv does not officially support Windows, therefore you should instead ensure you have the correct version Python 3.9.6

## Installation
* For first time users in the root directory run:
    1. `poetry install` to install all depenencies
    2. `pyenv install` to install and use the correct version of python (for MacOS users)

## Running the code 
* To run the train script locally run `poetry run python src/train.py`
* Note: if you're just wanting to see the code in action, changing the number of epochs parameter in `train.py` will reduce the training time, enabling you to run the code quickly.
* To run the inference script locally:
    1. Make sure you have a model file stored as `model/model.h5`
    2. Run `poetry run python src/run.py`

## Docker 
To run the docker image:
1. Make sure you have a Docker daemon running (e.g. using Docker Desktop)
2. Run `docker build -t catch .` to build the dockerfile into an image with the tag `catch`
2. Run `docker run catch` to run the training image in a container. (add the `--rm` flag to delete the container after it has run)

## Adjusting Training Parameters
To explore and adjust the model training parameters, you can set the environment variables:
- `TRAIN_EPOCHS`
- `TRAIN_EPSILON`
- `TRAIN_MAX_MEMORY`
- `TRAIN_HIDDEN_SIZE`
- `TRAIN_HIDDEN_LAYERS`
- `TRAIN_BATCH_SIZE`
- `TRAIN_GRID_SIZE` 

In addition, if you want to warm start the model, set `TRAIN_WARM_START_PATH` with a previous model's weights file. For example `./model/model.h5`

### Direnv
[Direnv](https://direnv.net) is a great tool to define environment variables at a folder level.  
To setup direnv, install the tool and create a `.envrc` file with environment variables to define.  
`.envrc.sample` is a sample direnv config file which can be copied.

To initiate direnv on the repo, run command: 
```
direnv allow
```

# Contributing
If you would like to develop on this repo ensure that the pre-commit hooks run on your local machine. To enable this run:
```
pip install pre-commit
pre-commit install
```
