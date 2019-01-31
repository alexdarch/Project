from gym.envs.classic_control import CartPoleEnv  # CartPoleEnv is a module, not an attribute -> can't indirectly import
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# Ought to rewrite so that state2D and state1D are attributes? Maybe consider using getter/setter methods
class CartPoleWrapper(CartPoleEnv):
    """
    This wrapper changes/adds:
      - 'steps_beyond_done' now can go any number of steps beyond done (for calculating state values)
      - 'step' now returns a loss equivalent to the squared error of position
      - 'state_loss' returns the loss for the state for use in 'step'
      - 'reset' functionality to be able to set the start state by passing [x_pos, x_vel, angle, ang_vel]

      - 'get_state_2d' converts an observation into a 2D state representation, taking into account previous states
      - 'print_state_2d' shows an image of the 2D state
      - 'get_state_1d' converts state_2d into a 1d representation so the MCTS can store it

      - 'get_action' (redundant) converts an action of 1 / 0 to a list: [0, 1] or [1, 0]
      - 'get_state_2d_size', 'get_action_size', 'get_observation', 'get_state_2d_size' are obvious
    """

    def __init__(self):
        super().__init__()
        self.mcts_bins = 25

        # state_2d parameters
        self.discount = 0.7
        self.pos_size = 20
        self.ang_size = 20

        # to stop episodes running over
        self.steps = 0
        self.max_steps_beyond_done = 16
        self.max_till_done = 200

    def step(self, action):
        observation, reward, done, info = super().step(action)
        loss = self.state_loss()
        self.steps += 1

        # make sure that we cant go further than 200 steps
        if self.steps >= self.max_till_done:
            done = True

        return observation, loss, done, info

    def reset(self, init_state=None):
        rand_obs = super().reset()  # call the base reset to do all of the other stuff
        self.steps = 0
        self.steps_beyond_done = -self.max_steps_beyond_done  # counts up to 0 from -16

        if init_state is None:
            return rand_obs
        self.state = np.array(init_state)  # and then edit state if we want (state is a base class attribute)
        return self.state

    def get_state_2d(self, curr_state, prev_state_2d=None):
        """
        curr_state numpy array [x_pos, x_vel, angle, ang_vel]
        even though it says state[3] = tip_vel in the docs, its actually ang_vel (since length = 1)
        max values are:        [+-2.4, inf, +-12 = +-0.21rad, inf]
        """
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
        done = abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold_radians
        # g = np.random.randint(0, high=2)
        # if g == 0:
        #     g = -1
        if done:
            return -1   # -1 is the maximum loss possible

        return -0.5*((self.state[0]/self.x_threshold) ** 2 + (self.state[2]/self.theta_threshold_radians) ** 2)
        # return -0.5 * ((self.state[0] / self.x_threshold) + (self.state[2] / self.theta_threshold_radians))

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

    def get_rounded_observation(self):
        # get the values to be roughly within +-1
        obs = [self.state[0]/self.x_threshold,
               self.state[1]/self.x_threshold,
               self.state[2]/self.theta_threshold_radians,
               self.state[3]/self.theta_threshold_radians
               ]
        # what about calculating a list of bin edges at the start using inverse cfd (norm.ppf),
        # and then running each thing through until we get to said list idx then returning the index as the
        # norm cfd'd output? would remove the whole astype(int) things too so maybe even faster if small enough?

        # obs = norm.cdf(obs, scale=1/3)  # want +-1 to lie on the +-3std point -> scale down by 1/3
        # obs = np.rint(obs * self.mcts_bins).astype(int)  # norm.cdf returns np.array(), if don't round then 3.8 -> 3
        # return tuple(obs.tolist())

        obs = [int(round(elm*self.mcts_bins)) for elm in obs]
        return tuple(obs)

    def get_action_size(self):  # only works for discrete actions need to update!
        x = 0
        while self.action_space.contains(x):
            x += 1
        return x

    def get_state_2d_size(self):
        return self.pos_size, self.ang_size

    def print_state_2d(self, state_2d):
        plt.imshow(state_2d, extent=[-self.x_threshold, self.x_threshold, -self.theta_threshold_radians,
                                     self.theta_threshold_radians], cmap='jet', aspect='auto')
        plt.xlabel("X-Position")
        plt.ylabel("Angular Position")
        plt.colorbar()
        plt.show()

# env = CartPoleWrapper()
# print(env.get_action(1))



