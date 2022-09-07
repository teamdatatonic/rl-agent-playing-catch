"""
Catch Game Training Script

Python file for running the Q-learning training process for game of Catch.
Executed using 'poetry run python train.py'.

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""

import json
import numpy as np
import logging
import os
from typing import Type

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import tensorflow as tf
import keras

from experience_replay import ExperienceReplay
from env import Catch


def define_model(
    hidden_size: int,
    num_actions: int,
    learning_rate: float = 0.1,
    hidden_activation: str = "relu",
    loss: str = "mse",
    hidden_layers: int = 2,
) -> Type[keras.Model]:
    model = Sequential()
    model.add(
        Dense(hidden_size, input_shape=(grid_size**2,), activation=hidden_activation)
    )
    # Dynamically add additional hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_size, activation=hidden_activation))
    model.add(Dense(num_actions))
    model.compile(SGD(learning_rate=learning_rate), loss)
    return model


def train_model(
    model: Type[keras.Model],
    epochs: int,
    experience_replay: int,
    epsilon: int,
    batch_size: int,
) -> Type[keras.Model]:

    logging.info("Initializing model training")
    # Training loop
    win_count = 0
    for epoch in range(epochs):
        # Reset and get initial input
        env.reset()
        current_state = env.observe()

        loss = 0.0
        game_over = False

        while not game_over:
            # As we move into next step current state becomes previous state
            previous_state = current_state

            # Decide if next action is explorative or exploits current policy
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(previous_state)
                action = np.argmax(q[0])

            # Apply action and save rewards and new state
            current_state, reward, game_over = env.act(action)
            if reward == 1:
                win_count += 1

            # Store experience in experience replay
            experience_replay.add_experience(
                [previous_state, action, reward, current_state], game_over
            )

            # Adapt model
            inputs, targets = experience_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        logging.info(
            f"Epochs {np.round(epoch, 3)}/{epochs - 1} | Loss {np.round(loss, 4)} | Win count {win_count}"
        )
    return model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    logging.info(
        f"Number GPUs available: {len(tf.config.experimental.list_physical_devices('GPU'))}"
    )

    # Create all necessary folders
    model_path = "./model"
    os.makedirs(model_path, exist_ok=True)

    ## Define Parameters
    # You may want to use smaller parameter values, especially for the number of epochs if you want to see the code in action
    num_actions = 3  # [move_left, stay, move_right]

    # Environment variable parameters. Reduce to reduce training time.
    epochs = int(os.environ.get("TRAIN_EPOCHS", 3))
    epsilon = float(os.environ.get("TRAIN_EPSILON", 0.1))
    max_memory = int(os.environ.get("TRAIN_MAX_MEMORY", 640))
    hidden_size = int(os.environ.get("TRAIN_HIDDEN_SIZE", 100))
    hidden_layers = int(os.environ.get("TRAIN_HIDDEN_LAYERS", 2))
    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", 64))
    grid_size = int(os.environ.get("TRAIN_GRID_SIZE", 10))
    warm_start_model = os.environ.get("TRAIN_WARM_START_PATH")

    ## Define Model
    model = define_model(hidden_size, num_actions, hidden_layers=hidden_layers)

    # If you want to continue training from a previous model.
    if warm_start_model:
        logging.info(f"Warm starting model with previous weights: {warm_start_model}")
        model.load_weights(warm_start_model)

    ## Define Environment
    env = Catch(grid_size)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    ## Train Model
    trained_model = train_model(model, epochs, exp_replay, epsilon, batch_size)

    # Save trained model weights and architecture, this will be used by the visualization code
    trained_model.save_weights(f"{model_path}/model.h5", overwrite=True)
    with open(f"{model_path}/model.json", "w") as outfile:
        json.dump(trained_model.to_json(), outfile)
    logging.info(f"Model saved to path {model_path}")
