"""Experience Replay Class Definition

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""

import numpy as np


class ExperienceReplay(object):
    """A class to provide experience replay.
    Contains the inialization of the replay buffer with given parameters, a method to remember
    current state and a method to randomly select a batch of experiences for training
    the model.
    """

    def __init__(self, max_memory=100, discount=0.9):
        """Initialization of the experience buffer.

        Args:
            max_memory (int, optional): Maximum length of experience lookback saved. Defaults to 100.
            discount (float, optional): Discount factor applied to Q-value look ahead. Defaults to 0.9.
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        """Memorize the current state of the game in the experience buffer.
        If max memory is reached delete oldest entry.

        Args:
            states (List): [previous_state, action, reward, current_state]
            game_over (bool): Is game over True or False
        """
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        """Get batch of inputs and target values, given a specified batch_size.

        Args:
            model (tf.keras.Model): Sequential neural network for predicting Q-values.
            batch_size (int, optional): Batch size of inputs and targets to return. Defaults to 10.

        Returns:
            inputs (np.array), targets (np.array): state inpus to model, q-value from model for given state
        """
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, index in enumerate(
            np.random.randint(0, len_memory, size=inputs.shape[0])
        ):
            previous_state, action_t, reward, current_state = self.memory[index][0]
            game_over = self.memory[index][1]

            inputs[i] = previous_state
            targets[i] = model.predict(previous_state)[0]
            Q_sa = np.max(model.predict(current_state)[0])

            if game_over:  # If game_over is True
                targets[i, action_t] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward + self.discount * Q_sa
        return inputs, targets
