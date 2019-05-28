from gym.envs.classic_control import CartPoleEnv  # CartPoleEnv is a module, not an attribute -> can't indirectly import
import numpy as np
from scipy.stats import norm
# from utils import *
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

    def __init__(self, adversary, unopposedTrains):
        super().__init__()

        # flattened and 2d state parameters
        self.state_bins = 40  # 30 puts us out of memory
        scaling = 1 / 3  # 1/3 gives a flat distribution if obs ~ gaussian, lowering it gives more weight to values -> 0
        # if we have linspace(0, 1, ..) then we get inf as the last two value -> they are never picked
        self.bin_edges = np.asarray(
            [norm.ppf(edge, scale=scaling) for edge in np.linspace(0.001, 0.999, num=self.state_bins)])
        self.discount = 0.5

        # Actions, Reset range and loss weighting:
        self.adversary = adversary  # Enumerate: 0:Adversary, 1:None 2:EveryStepRandom, 3:LargeRandom, 4:Normal
        self.unopposedTrains = unopposedTrains
        self.player_action_space = [1, -1]
        self.avg_adv_force = 0.05  # Given by equating linear thetadot from a force

        # ---------- Initialise the Adversary -------------
        if self.adversary == 0:  # (training) policy adversary
            self.adv_action_space = [1, -1]  # default is nnet adversary for #0 and #2
            self.handicap = 0
        if self.adversary == 1:  # (test) policy adversary
            self.adv_action_space = [1, -1]
            self.handicap = self.avg_adv_force
        if self.adversary == 2:  # (test) random adversary
            self.adv_action_space = [1, -1]
            self.handicap = self.avg_adv_force
        if self.adversary == 3:  # (test) no adversary
            self.adv_action_space = [0]
            self.handicap = 0

        self.reset_rng = 0.3  # +-rng around 0, when the state is normed (so x=[-1, 1], theta=[-1, 1]....)
        self.loss_weights = [0.4, 0.1, 0.7, 1]  # multiply state by this to increase it's weighing compared to x
        self.weight_norm = sum(self.loss_weights)  # increase this to increase the effect of the terminal cost
        self.terminal_cost = -1

        # to stop episodes running over
        self.steps = 0
        self.mcts_steps = 0
        self.max_steps_beyond_done = 15
        self.extra_steps = 0  # counts up to max_steps once done is returned
        self.steps_till_done = 400

        self.tau = 0.01  # 0.02

    def increment_handicap(self, iter):
        # note the first 1-0.5**0 = 0 -> add one to it
        self.handicap = 0 if iter < self.unopposedTrains else self.avg_adv_force * (1-0.7**(iter-self.unopposedTrains+1))  # 0.5 -> (0, 0.5, 0.75...)
        print('Handicap: ', self.handicap)

    def step(self, a, agent, next_true_step=False):
        assert a in range(self.get_action_size(agent)), "%r (%s) invalid action" % (a,  type(a))
        state = self.state
        x, x_dot, theta, theta_dot = state

        if agent == 0:
            f_1 = self.player_action_space[a] * self.force_mag
            f_2 = 0
        else:
            f_1 = 0
            f_2 = self.handicap * self.adv_action_space[a] * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = sintheta * self.polemass_length * theta_dot**2
        xacc = (f_1 + f_2 * np.cos(2*theta) + temp - self.masspole*self.gravity*sintheta*costheta)\
            / (self.masscart + self.masspole * sintheta**2)
        thetaacc = (temp + f_1 + f_2 - self.total_mass*xacc)/(self.polemass_length * costheta)

        # euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        # check if done
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        loss = self.state_loss()

        if next_true_step:
            self.steps += 1
            if done or self.steps > self.steps_till_done:
                self.extra_steps += 1
                done = False
            if self.extra_steps > self.max_steps_beyond_done:
                done = True  # needed to find the value for the last step
        else:
            self.mcts_steps += 1  # mcts doesnt get stopped until the true step > 200+max_steps_beyond_done

        # if it isn't a true step then return done if fallen over
        # -> doesn't stop the sim, but does add -1 to v in MCTS
        return np.array(self.state), loss, done, {}

    def get_agent_action(self, a, agent):
        # agent is always nnet, and if adv == 1, then it is nnet too
        if agent == 0 or self.adversary == 0:
            return a
        if self.adversary == 1:
            return 0

        # else, return an action from the other agent types
        if self.adversary == 2:  # Every Step is random with equal probabilities
            return np.random.choice(range(len(self.adv_action_space)))
        if self.adversary == 3:  # Every N steps, we take a random large hit. 0 otherwise
            p_force = 0.5*self.avg_adv_force/1  # 1 is the power of the impulse as a proportion of self.force_mag
            action = np.random.choice(range(len(self.adv_action_space)), p=[p_force, p_force, 1-2*p_force])
            return action

    def reset(self, init_state=None, reset_rng=None):

        if init_state is None:
            rng = self.reset_rng if reset_rng is None else reset_rng

            normed_obs = self.np_random.uniform(low=-rng, high=rng, size=(4,))
            self.state = self.undo_normed_observation(normed_obs)
            self.steps = 0  # only want to reset steps if it is a true reset
            self.mcts_steps = 0
            self.extra_steps = 0
            self.steps_beyond_done = None
            return np.array(self.state)
        else:
            self.mcts_steps = self.steps

        self.state = np.array(init_state)  # and then edit state if we want (state is a base class attribute)
        return self.state

    def get_state_2d(self, prev_state_2d=None):
        """
        curr_state numpy array [x_pos, x_vel, angle, ang_vel]
        even though it says state[3] = tip_vel in the docs, its actually ang_vel (since length = 1)
        max values are:        [+-2.4, inf, +-12 = +-0.21rad, inf]
        """

        return np.array(self.state)

        norm_obs = self.get_normed_observation()
        # get the index of the nearest bin edge. Since bin edges near 0 are closer, weighting is better
        norm_obs = [np.abs(self.bin_edges - elm).argmin() for elm in norm_obs]  # bin the normed obs
        edges = np.linspace(-0.5, self.state_bins - 0.5, self.state_bins + 1)  # need extra to get [-0.5, ..., 24.5]
        new_pos, _, _ = np.histogram2d([norm_obs[2], ], [norm_obs[0], ], bins=(edges, edges))
        if prev_state_2d is None:
            prev_obs = (self.state[2] - self.state[3], self.state[0] - self.state[1])  # (prev_theta, prev_x)
            prev_obs = (prev_obs[0] / self.theta_threshold_radians, prev_obs[1] / self.x_threshold)

            prev_obs_binned = [np.abs(self.bin_edges - elm).argmin() for elm in prev_obs]
            prev_pos, _, _ = np.histogram2d([prev_obs_binned[0], ], [prev_obs_binned[1], ], bins=(edges, edges))
            return new_pos + self.discount * prev_pos
        else:
            x = new_pos + self.discount * prev_state_2d

            return x


    def state_loss(self, state=None):
        if state is None:
            state = self.state
        # change the reward to -(x^2+theta^2). Technically a loss now
        done = abs(state[0]) > self.x_threshold or abs(state[2]) > self.theta_threshold_radians
        if done:
            return self.terminal_cost  # -1 is the maximum loss possible (?)

        norm_obs = self.get_normed_observation(state)
        weighted_states = [-w*(s**2)/self.weight_norm for w, s in zip(self.loss_weights, norm_obs)]
        return sum(weighted_states)

    def get_state(self):
        return self.state

    def get_mcts_state(self, state, acc):
        return tuple([int(dim * acc) for dim in state])

    def round_state(self, state):
        return str([round(dim, 2) for dim in state])

    def get_normed_observation(self, state=None):
        if state is None:
            state = self.state
        # get the values to be roughly within +-1
        obs = [state[0] / self.x_threshold,
               state[1] / self.x_threshold,
               state[2] / self.theta_threshold_radians,
               state[3] / (self.theta_threshold_radians*10)  # when normed, this has a 10x wider range than others
               ]
        return obs

    def undo_normed_observation(self, obs):
        unnormed_obs = [obs[0] * self.x_threshold,
                       obs[1] * self.x_threshold,
                       obs[2] * self.theta_threshold_radians,
                       obs[3] * (self.theta_threshold_radians*10)  # when normed, this has a 10x wider range than others
                       ]
        return unnormed_obs

    def get_action_size(self, agent):  # only works for discrete actions need to update!
        a_space = len(self.player_action_space)if agent == 0 else len(self.adv_action_space)
        return a_space

    def get_state_2d_size(self):
        return 4, 1
        return self.state_bins, self.state_bins

# env = CartPoleWrapper()
# print(env.get_action(1))



