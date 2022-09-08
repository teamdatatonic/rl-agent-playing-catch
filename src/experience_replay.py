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

    def __init__(self, max_memory: int = 100, discount: float = 0.99) -> None:
        """
        Initialization of the experience buffer.

        Args:
            max_memory (int, optional): Maximum length of experience lookback saved. Defaults to 100.
            discount (float, optional): Discount factor applied to Q-value look ahead. Defaults to 0.9.
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def add_experience(self, sars: list, game_over: bool) -> None:
        """
        Memorize the current state of the game in the experience buffer.
        If max memory is reached delete oldest entry.

        Args:
            sars (List): [previous_state, action, reward, current_state]
            game_over (bool): Is game over True or False
        """
        self.memory.append([sars, game_over])
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
        num_inputs = inputs.shape[0]
        targets = np.zeros((num_inputs, number_of_actions))

        # For each element in the batch of size batch_size randomly select an experience
        # save it into inputs, and calculate the q-values for state to be saved into
        # targets.
        ids = np.random.choice(memory_length, size=num_inputs, replace=False)

        # Select random experience
        sars = list(zip(*[self.memory[id_][0] for id_ in ids]))
        previous_states, action_ts, rewards, current_states = (
            np.concatenate(e) if isinstance(e[0], np.ndarray) else np.stack(e)
            for e in sars
        )
        game_over = np.stack([self.memory[id_][1] for id_ in ids])

        # Use state to calculate q-values for each action
        targets = model.predict(previous_states, batch_size=64, verbose=False)

        # Greedily choose maximum q-value
        Q_sa = np.max(model.predict(current_states, batch_size=64, verbose=False), 1)

        # Save into targets
        # If game_over is True use end reward
        # If not use current reward + discounted q-value
        # reward + gamma * max_a' Q(s', a')
        # reward is always zero in non terminal state, but is added for generality
        targets[np.arange(num_inputs), action_ts] = (
            rewards + self.discount * Q_sa * ~game_over
        )

        return previous_states, targets
