from IterationData import IterationData
import numpy as np
import matplotlib.pyplot as plt
# get a list of all the colour strings in matplotlib so we can iterate through them
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colour_list = list(colors.keys())
# 9 base colours ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'aliceblue', 'antiquewhite', 'aqua'...


class VisualiseIteration:
    """
    Uses objects of IterationData, which contians methods to easily access data, summarised below:

    Attributes:
        self.episodes = number of episodes in an iteration
        self.nnet_losses = Dataframe with ['PlayerAction', 'AdversaryAction', 'Value', 'Total'] as the column headders
        self.episode_data = A Dictionary of episodes with keys ['EpisodeN'] where N is an int.
                Each episode's value is a pandas dataframe with columns:

    ['TrueValue', 'PolicyValue', 'MCTSAction', 'PolicyAction', 'MCTSAdv', 'PolicyAdv',  'Observation',      'State2D']
      [ float,      float,         [f, f],        [f, f],      [f, f],     [f, f],  [x, xdot, theta, thetadot], 'arr_n']

        self.all_data = all of the episode data joined in columns as above

    Methods:
        get_state_2d(self, episode, step)
        get_episode_length(self, episode)
        get_max_episode(self)
    """

    def __init__(self, folder_path, iteration):

        self.iter = iteration
        self.action_space = [-1, 1]  # [left, right]
        self.action_names = {0: 'Left', 1: 'Right'}
        self.default_action = 0  # left is default action to show (no need to show left and right)
        self.iter_data = IterationData(folder_path, iteration)

        self.x_threshold = self.iter_data.x_threshold
        self.theta_threshold_radians = self.iter_data.theta_threshold_radians

    def add_axis_valuevsstep(self, episode=None, colour='blue', axes=None):
        # extract data for values and actions
        if episode is None:
            episode = self.iter_data.get_max_episode()
        policy_values = self.iter_data.episode_data['Episode' + str(episode)].PolicyValue
        true_values = self.iter_data.episode_data['Episode' + str(episode)].TrueValue

        # check if subplots already exist
        if axes is None:
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(17, 7))

        # Value Subplot
        axes.plot(policy_values, color=colour, linestyle='--',
                  label='Policy Values, Iter:' + str(self.iter) + ' Ep:' + str(episode))
        axes.plot(true_values, color=colour, linestyle='-',
                  label='True Values, Iter:' + str(self.iter) + ' Ep:' + str(episode))
        axes.legend()
        axes.set_ylabel('State Value')

        return axes

    def add_axis_actionvsstep(self, episode=None, agent='player', colour='blue', axes=None):

        assert agent == 'player' or agent == 'adversary' or agent == 'both', 'Not a valid agent'
        if episode is None:
            episode = self.iter_data.get_max_episode()

        if agent == 'player' or agent == 'both':
            policy_actions = self.iter_data.episode_data['Episode' + str(episode)].PolicyAction.values
            mcts_actions = self.iter_data.episode_data['Episode' + str(episode)].MCTSAction.values
        else:
            policy_actions = self.iter_data.episode_data['Episode' + str(episode)].PolicyAdv.values
            mcts_actions = self.iter_data.episode_data['Episode' + str(episode)].MCTSAdv.values

        policy_actions = list(zip(*policy_actions))[self.default_action]
        mcts_actions = list(zip(*mcts_actions))[self.default_action]

        # check if subplots already exist
        rows = 2 if agent == 'both' else 1
        if axes is None:
            fig, axes = plt.subplots(ncols=1, nrows=rows, figsize=(17, rows * 7))
        axes1 = axes[0] if agent == 'both' else axes

        axes1.plot(policy_actions, color=colour, linestyle='--',
                   label='Policy {:s} Actions, Iter: {} Ep: {}'.format(agent, self.iter, episode))
        axes1.plot(mcts_actions, color=colour, linestyle='-',
                   label='MCTS {:s} Actions, Iter: {} Ep: {}'.format(agent, self.iter, episode))
        axes1.legend()
        axes1.set_ylabel(self.action_names[self.default_action] + ' probability')
        axes1.set_xlabel('Steps')

        if agent == 'both':
            axes2 = axes[1]
            policy_adv = self.iter_data.episode_data['Episode' + str(episode)].PolicyAdv.values
            mcts_adv = self.iter_data.episode_data['Episode' + str(episode)].MCTSAdv.values

            policy_adv = list(zip(*policy_adv))[self.default_action]
            mcts_adv = list(zip(*mcts_adv))[self.default_action]

            axes2.plot(policy_adv, color=colour, linestyle='--',
                       label='Policy {} Actions, Iter: {} Ep: {}'.format(agent, self.iter, episode))
            axes2.plot(mcts_adv, color=colour, linestyle='-',
                       label='MCTS {} Actions, Iter: {} Ep: {}'.format(agent, self.iter, episode))
            axes2.legend()
            axes2.set_ylabel(self.action_names[self.default_action] + ' probability')
            axes2.set_xlabel('Steps')
            axes1 = [axes1, axes2]

        return axes1

    def get_episode_value_stats(self):
        averages = -1 * np.ones((200 + 1,))
        deviations = np.zeros((200 + 1,))
        all_episodes = []
        # print(averages.head())
        for ep in range(self.iter_data.episodes):
            true_values = self.iter_data.episode_data['Episode' + str(ep)].TrueValue.values  # TrueValue or PolicyValue
            all_episodes.append(true_values)
            x = self.iter_data.get_episode_length(ep) + 1
            averages[0:x] += true_values
            averages[x:] += -1
            deviations[0:x] += true_values ** 2
            deviations[x:] += 1

        averages = averages / self.iter_data.episodes
        deviations = np.sqrt(deviations - averages ** 2) / self.iter_data.episodes  # to get the standard deviation
        return averages, deviations, all_episodes

    def plot_observations_histogram(self, normalised=True):

        all_observations = self.iter_data.all_data['Observation'].values

        # normalise all the observations
        if normalised:
            all_normed_observations = []
            for step in all_observations:
                obs = [step[0] / self.x_threshold,
                       step[1] / self.x_threshold,
                       step[2] / self.theta_threshold_radians,
                       step[3] / self.theta_threshold_radians,
                       ]
                all_normed_observations.append(obs)
            all_observations = all_normed_observations
        # flip the dimensions
        all_observations = list(zip(*all_observations))

        # Plot histograms
        fig, axes = plt.subplots(ncols=len(all_observations), nrows=1, figsize=(len(all_observations) * 5, 5))
        obs_label = ['x', '$\dot{x}$', '$\\theta$', '$\dot{\\theta}$']
        axes[0].set_ylabel('Probability of Observation')
        for dim in range(len(all_observations)):
            axes[dim].hist(all_observations[dim], bins=20, density=True, color=colour_list[dim], alpha=0.9)
            if normalised:
                axes[dim].set_xlabel('Normalised ' + obs_label[dim])
            else:
                axes[dim].set_xlabel(obs_label[dim])

    def get_actionvsstate_probs(self, policy, x_dot_fixed=0, theta_dot_fixed=0, bin_proportion=0.15):

        shape = self.iter_data.get_state_2d(episode=0, step=0).shape
        x_dot_bin_size = bin_proportion  # /self.x_threshold # either side of the value
        theta_dot_bin_size = bin_proportion  # /self.theta_threshold_radians # either side of the value

        # only 2 actions currently
        player_probs = np.zeros(shape, dtype=float)
        player_counter = np.zeros(shape, dtype=float)
        adv_probs = np.zeros(shape, dtype=float)
        adv_counter = np.zeros(shape, dtype=float)

        if policy:
            player_act = 'PolicyAction'
            adv_act = 'PolicyAdv'
        else:
            player_act = 'MCTSAction'
            adv_act = 'MCTSAdv'

        player_actions = list(zip(*self.iter_data.all_data[player_act].values))[self.default_action]
        adv_actions = list(zip(*self.iter_data.all_data[adv_act].values))[self.default_action]

        obs = list(zip(*self.all_data['Observation'].values))
        state_2ds = self.iter_data.all_data['State2D'].values
        obs0, obs1, obs2, obs3 = np.array(obs[0]), np.array(obs[1]), np.array(obs[2]), np.array(obs[3])

        is_near_x_dot = (obs1 > x_dot_fixed - x_dot_bin_size) & (obs1 < x_dot_fixed + x_dot_bin_size)
        is_near_theta_dot = (obs3 > theta_dot_fixed - theta_dot_bin_size) & (
                    obs3 < theta_dot_fixed + theta_dot_bin_size)
        filtered_player = np.array(player_actions)[is_near_x_dot & is_near_theta_dot]
        filtered_adv = np.array(adv_actions)[is_near_x_dot & is_near_theta_dot]
        filtered_states = np.array(state_2ds)[is_near_x_dot & is_near_theta_dot]

        for idx, step in enumerate(filtered_states):
            state_2d = self.state_2ds[step]
            row, column = np.unravel_index(np.argmax(state_2d, axis=None),
                                           shape)  # get the index of the current element

            player_probs[row, column] += filtered_player[idx]
            player_counter[row, column] += 1

            adv_probs[row, column] += filtered_adv[idx]
            adv_counter[row, column] += 1

        player_probs = np.true_divide(player_probs, player_counter)
        adv_probs = np.true_divide(adv_probs, adv_counter)
        return player_probs, adv_probs

    def add_axis_actionprobvsstate(self, prob_array, axes, agent='player', vs=(0.2, 0.8)):

        image = axes.imshow(prob_array, vmin=vs[0], vmax=vs[1],
                            extent=[-self.x_threshold, self.x_threshold, -self.theta_threshold_radians,
                                    self.theta_threshold_radians],
                            cmap='jet', aspect='auto')
        axes.set_xlabel("X-Position")
        axes.set_ylabel("Angular Position")
        axes.set_title('Probability of ' + agent + ' Choosing Left vs Position')

    def plot_state_2d(self, episode, step):
        # Plot MCTS action, action policy and values associated?
        fig = plt.figure()
        axes = fig.add_axes([0, 0, 1, 1])  # add an axis object to the figure
        state_2d = self.iter_data.get_state_2d(episode, step)

        # returns a colourAxisImage, that we need to map the colourbar to the figure
        mapable = axes.imshow(state_2d, extent=[-self.x_threshold, self.x_threshold, -self.theta_threshold_radians,
                                                self.theta_threshold_radians], cmap='jet', aspect='auto')
        axes.set_xlabel("X-Position")
        axes.set_ylabel("Angular Position")
        axes.set_title('The 2D State')