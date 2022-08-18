"""Catch Training Sript

Python file for running the Q-learning training process for game of Catch.
Executed using 'poetry run python train.py'.

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""

import json
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from agent import ExperienceReplay
from env import Catch

# Create all necessary folders
model_path = "./model/"
os.makedirs(model_path, exist_ok=True)

# Parameters
epsilon = 0.1  # exploration
num_actions = 3  # [move_left, stay, move_right]
epochs = 1000
max_memory = 500
hidden_size = 100
batch_size = 50
grid_size = 10

model = Sequential()
model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation="relu"))
model.add(Dense(hidden_size, activation="relu"))
model.add(Dense(num_actions))
model.compile(SGD(learning_rate=0.2), "mse")

# If you want to continue training from a previous model, just uncomment the line bellow
# model.load_weights("model.h5")

# Define environment/game
env = Catch(grid_size)

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)

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
        exp_replay.remember([previous_state, action, reward, current_state], game_over)

        # Adapt model
        inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

        loss += model.train_on_batch(inputs, targets)
    print(
        "Epochs {:03d}/{} | Loss {:.4f} | Win count {}".format(
            epoch, epochs - 1, loss, win_count
        )
    )

# Save trained model weights and architecture, this will be used by the visualization code
model.save_weights(model_path + "model.h5", overwrite=True)
with open(model_path + "model.json", "w") as outfile:
    json.dump(model.to_json(), outfile)
