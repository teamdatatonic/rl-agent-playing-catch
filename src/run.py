"""
Catch Game Inference Script

Python file for running the Catch game using a model trained in 'train.py'.
Executed using 'poetry run python run.py'.

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""
import imageio.v2 as imageio
import json
import keras
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Type
import os

from os import walk
from keras.models import model_from_json

from env import Catch


def _save_img(input: Type[np.array], image_path: str, c: int, grid_size: int) -> None:
    # Draw the environment
    plt.imshow(input.reshape((grid_size,) * 2), interpolation="none", cmap="gray")
    # Save an image of the environment
    plt.savefig(image_comp_path := f"{image_path}/{c}.png")
    logging.debug(f"Catch game image created: {image_comp_path}")


def run_game(
    model: Type[keras.Model], image_path: str, grid_size: int, game_iterations: int = 10
) -> None:
    c = 0

    # Define environment
    env = Catch(grid_size)

    # Run n iterations of the game
    for i in range(game_iterations):
        logging.info(f"Catch game iteration {i} starting")
        env.reset()
        game_over = False

        # Get initial input
        input_t = env.observe()
        _save_img(input_t, image_path, c, grid_size)
        c += 1

        while not game_over:
            # Get next action
            q = model.predict(input_t)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            _save_img(input_t, image_path, c, grid_size)
            c += 1


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Ensure folders are created
    model_path = "./model"
    image_path = "./image"
    gif_path = "./gif"

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(gif_path, exist_ok=True)

    # Make sure this grid size matches the value used for training
    grid_size = int(os.environ.get("TRAIN_GRID_SIZE", 10))
    game_iterations = int(os.environ.get("RUN_GAME_ITERATIONS", 10))

    # Load the trained model
    with open(model_comp_path := f"{model_path}/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights(f"{model_path}/model.h5")
    model.compile("sgd", "mse")
    logging.info(f"Model {model_comp_path} loaded and compiled")

    # play the game
    run_game(model, image_path, grid_size, game_iterations)

    # Generate Gif
    filenames = next(walk(image_path), (None, None, []))[2]
    images = []
    for filename in filenames:
        images.append(imageio.imread(f"{image_path}/{filename}"))
    imageio.mimsave(gif_comp_path := f"{gif_path}/catch.gif", images, fps=4)
    logging.info(f"Catch game GIF created at {gif_comp_path}")
