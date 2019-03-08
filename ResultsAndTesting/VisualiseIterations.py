import VisualiseIteration
from VisualiseIteration import VisualiseIteration
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# get a list of all the colour strings in matplotlib so we can iterate through them
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colour_list = list(colors.keys())
# 9 base colours ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'aliceblue', 'antiquewhite', 'aqua'...


class VisualiseIterations:
    """
    VisualiseIterations principally contains a list of VisualiseIteration objects, which in turn contain Iteration data objects
    leading to an objects hirachy of VisualiseIterations <- Iter*VisualiseIteration <- IterationData. The main components of
    the two attribute classes are shown below:

    VisualiseIterations:
    Attributes:
        self.iters_data = [iter_data0, iter_data1, iter_data2, ...]
        self.iterations

    VisualiseIteration:
    Attributes:
        self.iter, self.action_space = [-1, 1], self.action_names ={0: 'Left', 1: 'Right'}
        self.default_action = 0  # left is default action to show (no need to show left and right)
        iter_data = IterationData(folder_path, iteration)
    Methods:
        add_axis_valuevsstep(self, episode=None, colour = 'blue', axes=None)
        add_axis_actionvsstep(self, episode=None, agent='player', colour = 'blue', axes=None)
        get_episode_value_stats(self)
        plot_observations_histogram(self, normalised=True)
        get_actionvsstate_probs(self, policy, x_dot_fixed=0, theta_dot_fixed=0, bin_proportion=0.15)
        add_axis_actionprobvsstate(self, prob_array, axes, agent='player', vs=(0.2, 0.8))
        plot_state_2d(self, episode, step)

    IterationData:
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

    def __init__(self, data_set_path, iters='all'):
        self.folder_path = data_set_path

        self.iters_data = []
        if iters == 'all':
            self.iterations = 0
            while True:
                try:
                    self.iters_data.append(VisualiseIteration(self.folder_path, self.iterations))
                except:
                    break
                self.iterations += 1
        else:
            assert isinstance(iters, list)
            for itr in iters:
                self.iters_data.append(VisualiseIteration(self.folder_path, itr))
            self.iterations = len(self.iters_data)

        print("There are ", self.iterations, " iterations stored in self.iterations")

    def plot_valuevsstep(self, iters=(0, 1, 2), plot_stds=False, plot_all_eps=False):
        assert all(i < self.iterations for i in iters), "Some iterations are not valid!"

        fig = plt.figure(figsize=(17, 7))
        axes = fig.add_axes([0, 0, 1, 1])
        axes.set_xlabel('Steps')
        axes.set_ylabel('Average Value')
        axes.set_ylim([-1, 0])
        for itr in iters:
            means, std, all_eps = self.iters_data[itr].get_episode_value_stats()

            axes.plot(means, linestyle='-', color=colour_list[itr], label='Itr:' + str(itr) + ' Average Episode Value')
            if plot_stds:
                axes.plot(means + std, linestyle='--', color=colour_list[itr], label=None)
                axes.plot(means - std, linestyle='--', color=colour_list[itr], label=None)
            if plot_all_eps:
                for i in range(self.iters_data[itr].iter_data.episodes):
                    axes.plot(all_eps[i], linestyle=':', label=None)

            axes.legend()

    def plot_nnet_lossesvsbatch(self, iters=(0, 1, 2)):
        assert all(i < self.iterations for i in iters), "Some iterations are not valid!"

        fig = plt.figure(figsize=(17, 7))
        axes = fig.add_axes([0, 0, 1, 1])

        axes.set_xlabel('Batches (size 8)')
        axes.set_ylabel('Loss')

        for itr in iters:
            axes.plot(self.iters_data[itr].iter_data.nnet_losses)
            axes.legend(['Itr ' + str(itr) + ' Player Action', 'Itr ' + str(itr) + ' Adversary Action',
                         'Itr ' + str(itr) + ' Value', 'Itr ' + str(itr) + ' Total'])

    def plot_observation_histograms(self, iters=(0, 1, 2), normed=True):
        for itr in iters:
            self.iters_data[itr].histogram_observations(normed)

    def plot_valuesvssteps(self, iters=(0, 1, 2), episodes='longest'):
        # fig, ax = te0.plot_episode(episode=3, colour='blue')
        max_eps = []
        for itr in iters:
            if episodes == 'longest':
                max_eps.append(self.iters_data[itr].get_max_episode())
            if isinstance(episodes, list):
                pass

        # keep plotting iterations on the same axis
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(17, 7))
        for idx, itr in enumerate(iters):
            ax = self.iters_data[itr].add_axis_valuevsstep(max_eps[idx], itr, colour_list[idx], ax)

    def plot_actionsvssteps(self, iters=(0, 1, 2), episodes='longest'):
        max_eps = []
        for itr in iters:
            if episodes == 'longest':
                max_eps.append(self.iters_data[itr].get_max_episode())
            if isinstance(episodes, list):
                pass
        # plot those episodes, one for the player and one for the adversary
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(17, 2 * 7))
        for idx, itr in enumerate(iters):
            ax = self.iters_data[itr].add_axis_actionvsstep(max_eps[idx], 'player', colour_list[idx], ax)

    def plot_actionprobsvsstates(self, iters=(0, 1, 2), policy=True, x_dot_fixed=0, theta_dot_fixed=0, bin_prop=0.05,
                                 colourbar_lims=(0, 1)):
        assert all(i < self.iterations for i in iters), "Some iterations are not valid!"

        for itr in iters:
            player_probs, adversary_probs = self.iters_data[itr].get_actionvsstate_probs(policy, x_dot_fixed,
                                                                                         theta_dot_fixed, bin_prop)

            # Plot the things
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(17, 5), sharey=True)
            ax, im = self.iters_data[itr].add_axis_actionprobvsstate(player_probs, axes[0], agent='player',
                                                                     vs=colourbar_lims)
            ax, im = self.iters_data[itr].add_axis_actionprobvsstate(adversary_probs, axes[1], agent='adversary',
                                                                     vs=colourbar_lims)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

    def vs_state_3d(self, angle=45, iteration=0, data_set='PolicyAction', omitted_state='x_dot', omitted_state_value=0,
                    binsize=0.15):

        cmap = 'viridis' if data_set in ['PolicyAction', 'PolicyAdv', 'PolicyValue'] else 'plasma'
        colour_ttl = 'State Value' if data_set in ['PolicyValue',
                                                   'TrueValue'] else 'Probability of \n the Agent Pushing Left'
        non_omitted_state_idx = 1 if omitted_state == 'theta_dot' else 3
        itr = iteration

        try:
            # print(self.iters_data[itr], self.iters_data[itr].iter_data)
            action_or_value = list(zip(*self.iters_data[itr].iter_data.all_data[data_set].values))[0]
        except:
            action_or_value = self.iters_data[itr].iter_data.all_data[data_set].values
        obs = list(zip(*self.iters_data[itr].iter_data.all_data['Observation'].values))

        # set up the plot
        fig = plt.figure(figsize=(17, 14))
        ax = fig.add_subplot(111, projection='3d')
        if omitted_state == 'theta_dot':
            omitted_obs = np.array(obs[3])
            ax.set_ylabel('$\dot{x}$, x-velocity (m/s)')
            ax.set_title(
                'Iteration {} {} with $\dot{{\\theta}}$ = {}$\pm${} rad/s'.format(itr, data_set, omitted_state_value,
                                                                                  binsize))

        else:
            omitted_obs = np.array(obs[1])
            ax.set_ylabel('$\dot{\\theta}$, angular velocity (rad/s)')
            ax.set_title(
                'Iteration {} {} with $\dot{{x}}$ = {}$\pm${} m/s'.format(itr, data_set, omitted_state_value, binsize))

        x_min, x_max = min((obs[0])) - 1, max(obs[0])
        omitted_min, omitted_max = min(omitted_obs) - 1, max(omitted_obs)
        theta_min, theta_max = min(obs[2]) - 0.25, max(obs[2])

        # extract the relavent points
        is_near_filter = (omitted_obs > omitted_state_value - binsize) & (omitted_obs < omitted_state_value + binsize)
        x, y, z = np.array(obs[0])[is_near_filter], np.array(obs[non_omitted_state_idx])[is_near_filter], \
                  np.array(obs[2])[is_near_filter]
        density = np.array(action_or_value)[is_near_filter]

        # plot the main plot
        ax.scatter(x, y, z, c=density, alpha=1, cmap=cmap)
        ax.set_xlabel('x position (m)')
        ax.set_zlabel('$\\theta$ position (rad)')

        # and add the 2d compressed versions

        ax.scatter(x, y, theta_min, zdir='z', c=density, s=5, alpha=0.5, cmap=cmap)
        ax.scatter(x_min, y, z, zdir='z', c=density, s=5, alpha=0.5, cmap=cmap)
        ax.scatter(x, omitted_min, z, zdir='z', c=density, s=5, alpha=0.5, cmap=cmap)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(omitted_min, omitted_max)
        ax.set_zlim(theta_min, theta_max)

        # sort out the colourbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array(density)
        mappable.set_clim(min(action_or_value), max(action_or_value))
        plt.colorbar(mappable, cax=cbar_ax)
        cbar_ax.set_title(colour_ttl)

        ax.view_init(25, angle)
        plt.show()
