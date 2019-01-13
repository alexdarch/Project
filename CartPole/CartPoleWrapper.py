from gym.envs.classic_control import CartPoleEnv  # CartPoleEnv is a module, not an attribute -> can't indirectly import
import matplotlib.pyplot as plt
import numpy as np


# Ought to rewrite so that state2D and state1D are attributes?
# would need to change get2Dstate to just age getter & update step & reset. Move calculation 2D to a different method
# Also should change state2D and get2Dstate so they match
class CartPoleWrapper(CartPoleEnv):
    """
    This wrapper adds:
      - 'reached_goal' property, where goal is to keep the cartpole upright for 200 steps.
      - 'reset' functionality to be able to set the start state
    """

    def __init__(self):
        super().__init__()
        # self._max_steps = env.spec.max_episode_steps
        self.discount = 0.7
        self.pos_size = 20
        self.ang_size = 20
        self.steps_beyond_done = 16  # remember to change reset too
        self.pareto = self.x_threshold / self.theta_threshold_radians  # normalise so that x and theta are of the same magnitude

    def step(self, action):
        observation, reward, done, info = super().step(action)
        loss = self.state_loss()
        return observation, loss, done, info

    def reset(self, init_state=None):
        rand_obs = super().reset()  # call the base reset to do all of the other stuff
        self.steps_beyond_done = 16  # removes the warning
        if init_state is None:
            return rand_obs
        self.state = np.array(init_state)  # and then edit state if we want (state is a base class attribute)
        return self.state

    def get_state_2d(self, curr_state, prev_state_2d=None):
        # curr_state numpy array [x_pos, x_vel, angle, ang_vel]
        #  even though it says state[3] = tip_vel in the docs, its actually vel
        # max values are:        [+-2.4, inf, +-12 = +-0.21rad, inf]
        pos_edges = np.linspace(-self.x_threshold, self.x_threshold, self.pos_size + 1)
        ang_edges = np.linspace(-self.theta_threshold_radians, self.theta_threshold_radians, self.ang_size + 1)
        new_pos, _, _ = np.histogram2d([curr_state[2], ], [curr_state[0], ], bins=(ang_edges, pos_edges))
        prev_pos, _, _ = np.histogram2d([curr_state[2] - curr_state[3], ], [curr_state[0] - curr_state[1], ],
                                        bins=(ang_edges, pos_edges))
        if prev_state_2d is None:
            return new_pos + self.discount * prev_pos
        else:
            return new_pos + self.discount * prev_state_2d

    def state_loss(self):
        # change the reward to -(x^2+theta^2). Technically a loss now
        return -self.state[0] ** 2 - self.pareto * self.state[2] ** 2

    @staticmethod
    def get_state_1d(state_2d):
        state_1d = np.reshape(state_2d, (-1,))  # converts to (10000,) array
        # s_ten = torch.tensor(state_1d, dtype=torch.float)
        return tuple(state_1d.tolist())

    @staticmethod
    def get_action(action):
        assert (action == 0 or action == 1)

        if action == 0:
            return np.array([1, 0])
        return np.array([0, 1])

    def get_observation(self):
        return self.state

    def get_action_size(self):  # only works for discrete actions need to update!
        x = 0
        while self.action_space.contains(x):
            x += 1
        return x

    def get_state_2d_size(self):
        return self.pos_size, self.ang_size

    def print_state(self, state_2d):
        plt.imshow(state_2d, extent=[-self.x_threshold, self.x_threshold, -self.theta_threshold_radians,
                                     self.theta_threshold_radians], cmap='jet', aspect='auto')
        plt.xlabel("X-Position")
        plt.ylabel("Angular Position")
        plt.colorbar()
        plt.show()

# env = CartPoleWrapper()
# print(env.get_action(1))



