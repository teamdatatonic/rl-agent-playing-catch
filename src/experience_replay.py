"""
Catch Game Experience Replay Class Definition

Python file containnig the Experience Replay class definitioin.

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""

import numpy as np
import keras
from typing import Tuple, Type


class ExperienceReplay(object):
    """
    A class to provide experience replay.
    Contains the initalization of the replay buffer with given parameters, a method to add_experience
    current state and a method to randomly select a batch of experiences for training
    the model.
    """

    def __init__(self, max_memory: int = 100, discount: int = 0.9) -> None:
        """
        Initialization of the experience buffer.

        Args:
            max_memory (int, optional): Maximum length of experience lookback saved. Defaults to 100.
            discount (float, optional): Discount factor applied to Q-value look ahead. Defaults to 0.9.
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def add_experience(self, states: list, game_over: bool) -> None:
        """
        Memorize the current state of the game in the experience buffer.
        If max memory is reached delete oldest entry.

        Args:
            states (List): [previous_state, action, reward, current_state]
            game_over (bool): Is game over True or False
        """
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(
        self, model: Type[keras.Model], batch_size: int = 10
    ) -> Tuple[np.array, np.array]:
        """
        Randomly selects a batch of experiences (of size batch_size) from the saved
        experiences in self.memory, and returns the state inputs for each experience,
        along with the q-values for each experience for every possible action (left, stay, right).

        Args:
            model (tf.keras.Model): Sequential neural network for predicting Q-values.
            batch_size (int, optional): Batch size of inputs and targets to return. Defaults to 10.

        Returns:
            inputs (np.array), targets (np.array): state inpus to model, q-value from model for given state
        """
        memory_length = len(self.memory)
        number_of_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]

        # Placeholder for state inputs saved in the experience buffer
        inputs = np.zeros((min(memory_length, batch_size), env_dim))

        # Placeholder for target q-values after applying model to state input
        targets = np.zeros((inputs.shape[0], number_of_actions))

        # For each element in the batch of size batch_size randomly select an experience
        # save it into inputs, and calculate the q-values for state to be saved into
        # targets.
        for i, index in enumerate(
            np.random.choice(memory_length, size=inputs.shape[0], replace=False)
        ):
            # Select random experience
            previous_state, action_t, reward, current_state = self.memory[index][0]
            game_over = self.memory[index][1]

            inputs[i] = previous_state

            # Use state to calculate q-values for each action
            targets[i] = model.predict(previous_state)[0]

            # Greedily choose maximum q-value
            Q_sa = np.max(model.predict(current_state)[0])

            # Save into targets
            # If game_over is True use end reward
            # If not use current reward + discounted q-value
            if game_over:
                targets[i, action_t] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward + self.discount * Q_sa

        return inputs, targets
