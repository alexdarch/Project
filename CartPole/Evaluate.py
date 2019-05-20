from MCTS import MCTS
import numpy as np
from utils import *
import pandas as pd
import time, csv, os
from utils import *


class Evaluate:
    def __init__(self, curr_policy, challenger_policy, env, args, curr_player_elo, curr_adv_elo):
        """
        Evaluates the current policy against the challenger policy. It pits the challenger player vs the current
        adversary and vice versa.
        Input:
            self.curr_action, self.challenger_action: functions that call get_action_prob(state_2d, root_state, agent, temp=0)
                                                      each has its own mcts and therefore policy.
                                             returns: player_probs, adversary_probs
            self.env: The current environment
        """
        # these are actually function pointers to getActionProb, that returns the greedy actions
        self.current_action = curr_policy
        self.challenger_action = challenger_policy
        self.env = env
        self.args = args
        self.tot_ep_length = self.env.steps_till_done + self.env.max_steps_beyond_done + 1

        self.chal_player_elo = curr_player_elo
        self.chal_adv_elo = curr_adv_elo

    def compare_policies(self, policies):
        num_policies = len(policies)  # always 2
        episode_costs = []
        episode_lengths = []
        for policy in range(num_policies):
            costs = -1 * np.ones((self.tot_ep_length,), dtype=float)
            observation = self.env.reset(reset_rng=0.05)
            counter, agent, done = 0, 0, False
            state_2d = self.env.get_state_2d()
            while not done:
                agent %= num_policies
                if self.args.renderTestEps:
                    self.env.render()
                    # time.sleep(self.env.tau)

                state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
                # alternate between the player taking an action and the adversary. After done, switch the policies
                if agent == 0:
                    a = policies[policy](state_2d, observation, agent)
                else:
                    a = policies[(policy+1) % num_policies](state_2d, observation, agent)

                observation, loss, done, info = self.env.step(a, agent, next_true_step=True)
                costs[counter] = loss
                counter += 1
                agent += 1
            episode_costs.append(costs)
            episode_lengths.append(counter)
        # episode scores are 0<s<1, with s high = good.
        ep_scores = [1+episode_costs[0], -episode_costs[1]]  # 1 for adversary is fallen over
        return ep_scores

    def evaluate_policies(self):
        """
        Compares self.challenger_nnet and self.nnet by comparing the means and medians of played episodes
        returns: update = True/False depending on whether some condition is met.
        """
        # the curr adv vs curr player scores should be 1-curr_s_player=curr_s_adv
        # The chal_s_player is chal_player vs curr_adv, and chal_s_adv is chal_adv vs curr_player
        curr_player_scores, curr_adv_scores = [], []
        chal_player_scores, chal_adv_scores = [], []
        # ------ PLAY EPISODES WITH CHALLENGER POLICIES ------------
        start = time.time()
        for n in range(self.args.testEps):

            # calculate the base scores
            curr_player_score, curr_adv_score = self.compare_policies([self.current_action, self.current_action])
            curr_player_scores.append(curr_player_score)
            curr_adv_scores.append(curr_adv_score)

            # calculate the scores of the challengers
            chal_player_score, chal_adv_score = self.compare_policies([self.challenger_action, self.current_action])
            chal_player_scores.append(chal_player_score)
            chal_adv_scores.append(chal_adv_score)

            Utils.update_progress("CHALLENGER POLICIES TEST EPISODES, ep" + str(n+1)+"(x4)",
                                  (n+1) / self.args.testEps, time.time() - start)
        self.env.close()
        # Compute the stats for the episode, dictionary of mean, var, episode lengths etc
        curr_player = self.compute_statistics(curr_player_scores, 'Player')
        curr_adv = self.compute_statistics(curr_adv_scores, 'Adversary')
        chal_player = self.compute_statistics(chal_player_scores, 'Player')
        chal_adv = self.compute_statistics(chal_adv_scores, 'Adversary')

        # and approximate the probability of the challenger beating the current policy with a logistic function
        chal_player['improvement'] = 400*np.log10(((1 - curr_player['score_mean'])*chal_player['score_mean'])/((1 - chal_player['score_mean']) * curr_player['score_mean']))
        chal_adv['improvement'] = 400*np.log10(((1 - curr_adv['score_mean'])*chal_adv['score_mean'])/((1 - chal_adv['score_mean']) * curr_adv['score_mean']))
        curr_player['improvement'], curr_adv['improvement'] = 0, 0
        print("player_improvement {}, adv_improvement {}".format(chal_player['improvement'], chal_adv['improvement']))

        # update elo's and record stats in dataframes
        curr_player['elo'], curr_adv['elo'] = self.chal_player_elo, self.chal_adv_elo
        self.chal_player_elo += chal_player['improvement']
        self.chal_adv_elo += chal_adv['improvement']
        chal_player['elo'], chal_adv['elo'] = self.chal_player_elo, self.chal_adv_elo

        self.save_to_csv([curr_player, curr_adv, chal_player, chal_adv])
        print("Elo's:", self.chal_player_elo, self.chal_adv_elo)

        return self.chal_player_elo, self.chal_adv_elo

    def compute_statistics(self, scores, agent='Player'):
        stats = {'score_mean': 0, 'score_var': 0, 'len_mean': 0, 'len_var': 0}
        # extract all data from scores
        all_scores = []
        all_lengths = []
        for episode in scores:
            all_scores.extend(episode)
            is_terminal = episode <= 0.0001 if agent == 'Player' else episode >= 0.9999
            all_lengths.append(len(episode[is_terminal == False]))  # extract the episode length

        # compute stats
        stats['len_mean'] = np.mean(all_lengths)
        stats['len_var'] = np.var(all_lengths, ddof=1)  # sample variance

        stats['score_mean'] = np.mean(all_scores)
        stats['score_var'] = np.var(all_scores, ddof=1)
        # save them all to csv
        # visualise them and then compare them in the report
        return stats

    def save_to_csv(self, data):
        file_name = ['curr_player', 'curr_adv', 'chal_player', 'chal_adv']
        folder = os.path.join('Data', 'TestData')
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)

        for idx, agent in enumerate(data):
            print(file_name[idx], ":  ", agent)
            df = pd.DataFrame(agent, index=[0])
            file_path = os.path.join(folder, file_name[idx] + '.csv')
            if not os.path.exists(file_path):
                with open(file_path, 'w+') as f:
                    df.to_csv(f, header=True)
            else:
                with open(file_path, 'a') as f:
                    df.to_csv(f, header=False)



