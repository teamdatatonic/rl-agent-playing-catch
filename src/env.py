"""Catch Environment Definition Class

Python file containnig the Catch class environment definition.

Authors: Sofie Verrewaere, Hiru Ranasinghe & Daniel Miskell @ Datatonic
"""

import numpy as np


class Catch(object):
    """
    Environment object for catch game, containing methods
    for initializing the game, updating state, displaying the current
    state of the game, as well as making observations and receiving
    rewards.
    """

    def __init__(self, grid_size=10):
        """Initialise grid with a defualt 10 by 10 size and reset upon construction.

        Args:
            grid_size (int, optional): Size of 2-dim grid_size by grid_size catch grid. Defaults to 10.
        """
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """Takes action given from (left, stay, right) and applies action to find next state
        of the fruit and basket.

        Args:
            action (int): Given action left (0), stay (1) or right (any other int).
        """
        # Get environment state
        state = self.state

        # Action is given as list idxs, convert to lateral action movements
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right

        # Decompose state into fruit position and basket position
        fruit_row, fruit_col, basket_position = state[0]

        # Update basket and fruit position
        new_basket_position = min(max(1, basket_position + action), self.grid_size - 2)
        fruit_row += 1
        out = np.asarray([fruit_row, fruit_col, new_basket_position])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _get_reward(self):
        """Use state of the game to return the current reward.

        Yield a reward of 1 if fruit is as bottom of canvas and is in contact with basket.
        Yield a reward of -1 if fruit is at the bottom of the canvas without contact with the basket.
        Otherwise 0 reward.

        Returns:
            int: Reward from current state (-1, 0, 1)
        """
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        """Game is over if fruit has reached bottom of canvas

        Returns:
            bool: True game is over, or False game isn't over.
        """
        if self.state[0, 0] == self.grid_size - 1:
            return True
        else:
            return False

    def observe(self):
        """Displays the state in a user friendly visualisation. Where the basket and fruit
        are 1's and everything else is 0's.

        Returns:
            np.array: 2-dim grid_size by grid_size of 0's and 1's
        """
        im_size = (self.grid_size,) * 2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # Draw fruit
        canvas[-1, state[2] - 1 : state[2] + 2] = 1  # Draw basket
        return canvas.reshape((1, -1))

    def act(self, action):
        """Given an action, update the state, get the rewards, check if it is over
         and redraw the canvas with the new state.

        Args:
            action (int):  Given action left (0), stay (1) or right (any other int).

        Returns:
            np.array, int, bool: Observation after action, reward of action, True of False
                whether game is the over.
        """
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        """Init the board with the fruit and basket starting at random positions."""
        n = np.random.randint(0, self.grid_size - 1, size=1)[0]
        m = np.random.randint(1, self.grid_size - 2, size=1)[0]
        self.state = np.asarray([0, n, m])[np.newaxis]
