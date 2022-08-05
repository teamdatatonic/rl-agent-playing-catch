import numpy as np

class Catch(object):
    # initialise grid and reset upon construction
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    # Given a state and action pair find the next state of the fruit and basket
    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
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
        
        # decompose state into fruit position and basket position
        fruit_row, fruit_col, basket_position = state[0]

        # Update basket and fruit position  
        new_basket_position = min(max(1, basket_position + action), self.grid_size-1)
        fruit_row += 1
        out = np.asarray([fruit_row, fruit_col, new_basket_position])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        # Only yield a reward if fruit is as bottom of canvas and is in contact with basket
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        # Game is over if fruit has reached bottom of canvas
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        # given an action, update the state, get the rewards, check if it is over and redraw the canvas with the new state
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        # Init the board with the fruit and basket starting at random positions
        n = np.random.randint(0, self.grid_size-1, size=1)[0]
        m = np.random.randint(1, self.grid_size-2, size=1)[0]
        self.state = np.asarray([0, n, m])[np.newaxis]

