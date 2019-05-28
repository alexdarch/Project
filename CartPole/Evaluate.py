# import gym
import numpy as np
from utils import *
from MCTS import MCTS
from utils import *
import pandas as pd
import time, csv, os


class Evaluate:
    def __init__(self, nnet, env, args):
        self.env = env
        self.tot_ep_length = self.env.steps_till_done + self.env.max_steps_beyond_done + 1
        self.pnet = nnet  # initialise player net (these are just place holders for loading in other ones)
        self.anet = self.pnet.__class__(self.env)  # initialise an adversary network
        self.args = args
        self.nnet_folder = os.path.join(self.args.policy_folder, self.args.checkpoint_folder)
        self.results_folder = os.path.join(self.args.policy_folder, self.args.results_folder)

        self.iteration = 0

    def evaluate(self, file):
        self.iteration = 0
        nnetname = 'checkpoint_' + str(self.iteration) + '.pth.tar'
        while os.path.exists(self.nnet_folder + '/player'+nnetname):
            if self.args.policy_folder is not None:
                # currently the player is always the neural network
                self.pnet.load_net_architecture(folder=self.nnet_folder, filename=nnetname)

            if self.args.adversary == 1:
                if self.args.adversaryIter is not None:
                    # if want to check a single iteration
                    self.anet.load_net_architecture(folder=self.nnet_folder, filename=nnetname)
                else:
                    # if want to see how it played against itself
                    self.anet.load_net_architecture(folder=self.nnet_folder, filename=nnetname)

            stats = pd.DataFrame(self.compare_policies(), index=[self.iteration])  # index for scalar values
            print(stats)
            self.save_to_csv(data=stats, folder=self.results_folder, filename=file)
            self.iteration += 1
            nnetname = 'checkpoint_' + str(self.iteration) + '.pth.tar'  # load_net loads both player and adversary nets

    def compare_policies(self):
        all_scores = []
        start = time.time()
        pmcts = MCTS(self.env, self.pnet, self.args)  # Player's MCTS tree
        for n in range(self.args.testEps):
            # define the agents
            player_action = lambda s2d, root, agent: np.argmax(pmcts.get_action_prob(s2d, root, agent, temp=0))
            adv_action = self.get_adversary_action()  # this returns a function pointer that takes (s2d, root, agent)

            # and execute an episode with them
            ep_scores = self.execute_episode([player_action, adv_action])
            all_scores.append(ep_scores)
            Utils.update_progress("TEST EPISODES, ADV{}. ITER {}, ep {}".format(self.args.adversary, self.iteration, n+1),
                                  (n+1) / self.args.testEps, time.time() - start)
            pmcts = MCTS(self.env, self.pnet, self.args)
        self.env.close()
        stats = self.compute_statistics(all_scores)
        return stats

    def execute_episode(self, get_action):
        # agents are functions [player(), adv()] that return their actions
        assert len(get_action) == 2, 'Incorrect number of agents'
        costs = -1 * np.ones((self.tot_ep_length,), dtype=float)
        counter, agent, done = 0, 0, False
        # ---------- Play Episode -------------
        observation = self.env.reset(reset_rng=0.05)
        state_2d = self.env.get_state_2d()
        while not done:
            agent %= 2
            if self.args.renderTestEps:
                self.env.render()

            state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
            # alternate between the player taking an action and the adversary.
            a = get_action[agent](state_2d, observation, agent)
            print('Agent: ', agent, '. Action and Pred: ', a, self.pnet.predict(state_2d, agent))
            observation, loss, done, info = self.env.step(a, agent, next_true_step=True)

            costs[counter] = loss
            counter += 1
            agent += 1

        ep_scores = 1+costs  # episode scores are 0<s<1, with s high = good
        return ep_scores

    def compute_statistics(self, scores):
        stats = {'score_mean': 0, 'score_std': 0, 'len_mean': 0, 'len_std': 0}
        all_scores = []
        all_lengths = []
        for episode in scores:
            all_scores.extend(episode)
            is_terminal = episode <= 0.0001
            all_lengths.append(len(episode[is_terminal == False]))  # extract the episode length
        # compute stats
        stats['len_mean'] = np.mean(all_lengths)
        stats['len_std'] = np.std(all_lengths, ddof=1)  # sample variance
        stats['score_mean'] = np.mean(all_scores)
        stats['score_std'] = np.std(all_scores, ddof=1)
        return stats

    def get_adversary_action(self):

        assert self.args.adversary != 0, 'Cannot use the training adversary for testing'

        if self.args.adversary == 1:
            amcts = MCTS(self.env, self.anet, self.args)
            return lambda s2d, root, agent: np.argmax(amcts.get_action_prob(s2d, root, agent, temp=0))

        if self.args.adversary == 2:
            return lambda s2d, root, agent: self.random_adversary(s2d, root, agent)
        if self.args.adversary == 3:
            return lambda s2d, root, agent: self.no_adversary(s2d, root, agent)
        print('This adversary has not been added yet')

    def random_adversary(self, s2d, root, agent):
        c = self.env.get_action_size(agent)
        return int(np.random.randint(0, c))  # low inclusive, high (c) exclusive

    def no_adversary(self):
        pass

    def save_to_csv(self, data, folder, filename):
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        file_path = os.path.join(folder, filename)

        if not os.path.exists(file_path+'.csv'):
            with open(file_path+'.csv', 'w+') as f:
                data.to_csv(f, header=True)
        else:
            with open(file_path+'.csv', 'a') as f:  # append if already exists (otherwise need to wait for results)
                data.to_csv(f, header=False)

