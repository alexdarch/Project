import pandas as pd
import numpy as np


class IterationData:
    """
    A class to load in the data from each iteration to an object in a readable form for later visualisation
    Note:
        GOOD get_state_2d(episode, step)
        BAD state_2ds[\'arr_x\']
    """

    def __init__(self, folder_path, iteration):

        self.training_path = folder_path + 'TrainingExamples' + str(iteration)
        self.nnet_loss_path = folder_path + 'NNetLosses' + str(iteration)

        # data formatting
        self.decimal_places = 3
        self.csv_rows = 6
        self.action_space = [-1, 1]

        # data manipulation
        self.x_threshold = 2.4
        self.theta_threshold_radians = 0.21

        # define a useful class for training examples, so we can access dict memebers as .'key' rather than ['key']
        class ReturnClass(dict):
            def __getattr__(self, name):
                return self[name]

        # just get the number of episodes (need to do this before adding episodes to the dict annoyingly)
        with open(self.training_path + '.csv', 'r', newline='') as f:
            row_count = sum([1 for row in f])
        self.episodes = int(row_count / self.csv_rows)

        # read in training examples data and nnet loss data
        # dict of {ep1: dataframe[2Dstate, TrueValue, PolicyValue, MCTSAction, PolicyAction, Observation], ep2: ...}
        self.episode_data = ReturnClass()
        self.all_data = pd.DataFrame(
            columns=['TrueValue', 'PolicyValue', 'MCTSAction', 'PolicyAction', 'Player', 'Observation',
                     'State2D'])
        self.state_2ds = np.load(self.training_path + '.npz')
        self.read_examples()
        self.nnet_losses = self.read_nnet_losses()

    def read_examples(self):
        step_counter = 0
        for episode in range(self.episodes):
            rows = [self.csv_rows * episode + x for x in range(self.csv_rows)]

            # only pick out relevent rows (because others are different lengths -> can't be read)
            episode_data = pd.read_csv(self.training_path + '.csv', header=None,
                                       skiprows=lambda x: x not in rows).transpose()
            episode_data.columns = ['TrueValue', 'PolicyValue', 'MCTSAction', 'PolicyAction', 'Player',
                                    'Observation']
            episode_data.TrueValue = episode_data.TrueValue.astype(float).round(self.decimal_places)
            episode_data.PolicyValue = episode_data.PolicyValue.astype(float).round(self.decimal_places)
            episode_data.Player = episode_data.Player.astype(int)

            # Split arrays and convert to floats
            episode_data.MCTSAction = episode_data.MCTSAction.apply(lambda x: x[1:-1].split(', '))
            episode_data.MCTSAction = episode_data.MCTSAction.apply(
                lambda x: [round(float(elm), self.decimal_places) for elm in x])

            episode_data.PolicyAction = episode_data.PolicyAction.apply(lambda x: x[1:-1].split(', '))
            episode_data.PolicyAction = episode_data.PolicyAction.apply(
                lambda x: [round(float(elm), self.decimal_places) for elm in x])

            episode_data.Observation = episode_data.Observation.apply(lambda x: x[1:-1].split())
            episode_data.Observation = episode_data.Observation.apply(
                lambda x: [round(float(elm), self.decimal_places) for elm in x])

            # add a reference to the stated2D for each step
            episode_states = list(self.state_2ds)[step_counter:step_counter + episode_data.index[-1] + 1]
            step_counter += episode_data.index[-1] + 1
            episode_data['State2D'] = episode_states

            # and then add the whole dataframe to the dict
            self.episode_data['Episode' + str(episode)] = episode_data
            self.all_data = pd.concat([self.all_data, episode_data], ignore_index=True)

    def read_nnet_losses(self):
        losses = pd.read_csv(self.nnet_loss_path + '.csv', header=None).transpose()
        losses.columns = ['PlayerAction', 'PlayerValue', 'AdversaryAction', 'AdversaryValue']
        return losses

    def get_state_2d(self, episode, step):
        episode = 'Episode' + str(episode)
        return self.state_2ds[self.episode_data[episode].State2D.values[step]]

    def get_episode_length(self, episode):
        episode = 'Episode' + str(episode)
        return self.episode_data[episode].index[-1]

    def get_max_episode(self):
        max_ep_len, max_ep = 0, 0
        for ep in range(self.episodes):
            ep_len = self.get_episode_length(ep)
            if ep_len > max_ep_len:
                max_ep_len = ep_len
                max_ep = ep
        return max_ep
