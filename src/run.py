"""Catch Game Sript

Python file for running the Catch game using a model trained in 'train.py'.
Executed using 'poetry run python run.py'.

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from os import walk
from keras.models import model_from_json

from env import Catch

# Ensure folders are created
model_path = "./model/"
image_path = "./image/"
gif_path = "./gif/"

os.makedirs(image_path, exist_ok=True)
os.makedirs(gif_path, exist_ok=True)

# Make sure this grid size matches the value used for training
grid_size = 10

# Load the trained model
with open(model_path + "model.json", "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights(model_path + "model.h5")
model.compile("sgd", "mse")

# Define environment, game
env = Catch(grid_size)
c = 0

# Run n iterations of the game
for _ in range(10):
    loss = 0.0
    env.reset()
    game_over = False
    # get initial input
    input_t = env.observe()

    # Draw the environment
    plt.imshow(input_t.reshape((grid_size,) * 2), interpolation="none", cmap="gray")
    plt.savefig(image_path + "%03d.png" % c)

    c += 1

    while not game_over:
        input_tm1 = input_t

        # get next action
        q = model.predict(input_tm1)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        input_t, reward, game_over = env.act(action)

        plt.imshow(input_t.reshape((grid_size,) * 2), interpolation="none", cmap="gray")
        plt.savefig(image_path + "%03d.png" % c)
        c += 1

# Generate Gif
filenames = next(walk(image_path), (None, None, []))[2]
images = []
for filename in filenames:
    images.append(imageio.imread(image_path + filename))
imageio.mimsave(gif_path + "catch.gif", images)
