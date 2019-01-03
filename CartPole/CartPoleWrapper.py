import gym
import matplotlib.pyplot as plt
import numpy as np


class CartPole(gym.core.Wrapper):
    """
    This wrapper adds 'reached_goal' property, where goal is to keep the cartpole upright for 200 steps.
    """

    def __init__(self, env):
        super(CartPole, self).__init__(env)
        self._steps = 0
        self._max_steps = env.spec.max_episode_steps
        self.discount = 0.7
        self.x_size = 100
        self.y_size = 100

    def reset(self):
        self._steps = 0
        return self.env.reset()

    def step(self, action):
        self._steps += 1
        return self.env.step(action)

    def get2Dstate(self, curr_state, prev2Dstate=None):
        # curr_state numpy array [xpos, xvel, angle, angle vel]
        # max values are:         [+-2.4, inf, +-41.8, inf]
        x_edges, y_edges = np.linspace(-1.2, 1.2, self.x_size + 1), np.linspace(-12 * np.pi / 180, 12 * np.pi / 180,
                                                                                self.y_size + 1)

        new_pos, _, _ = np.histogram2d([curr_state[0], ], [curr_state[2], ], bins=(x_edges, y_edges))
        prev_pos, _, _ = np.histogram2d([curr_state[0] - curr_state[1], ], [curr_state[2] - curr_state[3], ],
                                        bins=(x_edges, y_edges))

        if prev2Dstate is None:
            return new_pos + self.discount * prev_pos
        else:
            return new_pos + self.discount * prev2Dstate

    def get1Dstate(self, state_2D):
        traj1D = np.reshape(state_2D, (-1,))    # converts to (10000,) array
        # s_ten = torch.tensor(traj1D, dtype=torch.float)
        return tuple(traj1D.tolist())

    def getActionSize(self):    # only works for discrete actions need to update!
        x = 0
        while self.action_space.contains(x):
            x += 1
        return x

    def get2DstateSize(self):
        return self.x_size, self.y_size

    def printState(self, state_2D):
        plt.imshow(state_2D, extent=[-1.2, 1.2, -12 * np.pi / 180, 12 * np.pi / 180], cmap='jet', aspect='auto')
        plt.show()

    # @property
    # def reached_goal(self):
    #     return self._steps >= self._max_steps
