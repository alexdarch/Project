from gym.envs.classic_control import CartPoleEnv  # CartPoleEnv is a module, not an attribute -> can't indirectly import
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# from numba import jit, jitclass


class CartPoleWrapper(CartPoleEnv):
    """
    This wrapper changes/adds:
      - 'steps_beyond_done' now can go any number of steps beyond done (for calculating state values)
      - 'step' now returns a loss equivalent to the squared error of position
      - 'state_loss' returns the loss for the state for use in 'step'
      - 'reset' functionality to be able to set the start state by passing [x_pos, x_vel, angle, ang_vel]

      - 'get_state_2d' converts an observation into a 2D state representation, taking into account previous states
      - 'print_state_2d' shows an image of the 2D state
      - 'get_rounded_observation' bins observation so the MCTS can store it

      - 'get_action' (redundant) converts an action of 1 / 0 to a list: [0, 1] or [1, 0]
      - 'get_state_2d_size', 'get_action_size', 'get_observation', 'get_state_2d_size' are obvious
    """

    def __init__(self):
        super().__init__()
        # flattened and 2d state parameters
        self.state_bins = 25  # 30 puts us out of memory
        scaling = 1 / 3  # 1/3 gives a flat distribution if obs ~ gaussian, lowering it gives more weight to values -> 0
        # if we have linspace(0, 1, ..) then we get inf as the last two value -> they are never picked
        self.bin_edges = np.asarray(
            [norm.ppf(edge, scale=scaling) for edge in np.linspace(0.001, 0.999, num=self.state_bins)])
        self.discount = 0.5

        # to stop episodes running over
        self.steps = 0
        self.max_steps_beyond_done = 16
        self.extra_steps = 0  # counts up to max_steps once done is returned
        self.max_till_done = 200

    def step(self, action, next_true_step=False):
        observation, reward, done, info = super().step(action)
        loss = self.state_loss()

        if next_true_step:
            self.steps += 1
            if done or self.steps > self.max_till_done:
                self.extra_steps += 1
                done = False
            if self.extra_steps > self.max_steps_beyond_done:
                done = True
        # if it isn't a true step then return done if fallen over
        # -> doesn't stop the sim, but does add -1 to v in MCTS
        return observation, loss, done, info

    def reset(self, init_state=None):
        rand_obs = super().reset()  # call the base reset to do all of the other stuff
        self.steps_beyond_done = -1  # stops an error logging if we go beyond done

        if init_state is None:
            self.steps = 0  # only want to reset steps if it is a true reset
            self.extra_steps = 0
            return rand_obs

        self.state = np.array(init_state)  # and then edit state if we want (state is a base class attribute)
        return self.state

    def get_state_2d(self, prev_state_2d=None):
        """
        curr_state numpy array [x_pos, x_vel, angle, ang_vel]
        even though it says state[3] = tip_vel in the docs, its actually ang_vel (since length = 1)
        max values are:        [+-2.4, inf, +-12 = +-0.21rad, inf]
        """
        norm_obs = self.get_rounded_observation()
        edges = np.linspace(-0.5, self.state_bins-0.5, self.state_bins+1)  # need extra to get [-0.5, ..., 24.5]
        new_pos, _, _ = np.histogram2d([norm_obs[2], ], [norm_obs[0], ], bins=(edges, edges))

        if prev_state_2d is None:
            prev_obs = (self.state[2] - self.state[3], self.state[0] - self.state[1])    # (prev_theta, prev_x)
            prev_obs = (prev_obs[0] / self.theta_threshold_radians, prev_obs[1] / self.x_threshold)

            prev_obs_binned = [np.abs(self.bin_edges - elm).argmin() for elm in prev_obs]
            prev_pos, _, _ = np.histogram2d([prev_obs_binned[0], ], [prev_obs_binned[1], ], bins=(edges, edges))
            # prev_pos[prev_pos < 1 / (2 ** 5)] = 0   # only keep up to 4 times steps back
            return new_pos + self.discount * prev_pos
        else:
            # prev_state_2d[prev_state_2d < 1 / (2 ** 5)] = 0
            return new_pos + self.discount * prev_state_2d

    def state_loss(self):
        # change the reward to -(x^2+theta^2). Technically a loss now
        done = abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold_radians
        if done:
            return -1  # -1 is the maximum loss possible

        return -0.5 * ((self.state[0] / self.x_threshold) ** 2 + (self.state[2] / self.theta_threshold_radians) ** 2)

    @staticmethod
    def get_action(action):
        assert (action == 0 or action == 1)

        if action == 0:
            return np.array([1, 0])
        return np.array([0, 1])

    def get_observation(self):
        return self.state

    def get_rounded_observation(self):
        # get the values to be roughly within +-1
        obs = [self.state[0] / self.x_threshold,
               self.state[1] / self.x_threshold,
               self.state[2] / self.theta_threshold_radians,
               self.state[3] / self.theta_threshold_radians
               ]
        # get the index of teh nearest bin edge. Since bin edges near 0 are closer, weighting is better
        obs = [np.abs(self.bin_edges - elm).argmin() for elm in obs]  # should return the indexes of the norm.cdf
        return tuple(obs)

    def get_action_size(self):  # only works for discrete actions need to update!
        x = 0
        while self.action_space.contains(x):
            x += 1
        return x

    def get_state_2d_size(self):
        return self.state_bins, self.state_bins

    def print_state_2d(self, state_2d):
        plt.imshow(state_2d, extent=[-self.x_threshold, self.x_threshold, -self.theta_threshold_radians,
                                     self.theta_threshold_radians], cmap='jet', aspect='auto')
        plt.xlabel("X-Position")
        plt.ylabel("Angular Position")
        plt.colorbar()
        plt.show()

# env = CartPoleWrapper()
# print(env.get_action(1))



