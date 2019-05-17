from MCTS import MCTS
import numpy as np
from utils import *
import time, csv, os
from utils import *
EPS = 0 # 10e-5     # stop divide by zero error


class Evaluate:
    def __init__(self, curr_policy, challenger_policy, env, starting_player_elo, starting_adv_elo):
        """
        Evaluates the current policy against the challenger policy. It pits the challenger player vs the current
        adversary and vice versa.
        Input:
            self.curr_action, self.challenger_action: functions that call get_action_prob(state_2d, root_state, player, temp=0)
                                                      each has its own mcts and therefore policy.
                                             returns: player_probs, adversary_probs
            self.env: The current environment
        """
        # these are actually function pointers to getActionProb, that returns the greedy actions
        self.current_action = curr_policy
        self.challenger_action = challenger_policy
        self.env = env
        self.tot_ep_length = self.env.steps_till_done + self.env.max_steps_beyond_done + 1

        self.chal_player_elo = starting_player_elo
        self.chal_adv_elo = starting_adv_elo

    def compare_policies(self, policies, render=False):
        num_policies = len(policies)  # always 2
        episode_costs = []
        episode_lengths = []
        for policy in range(num_policies):
            costs = -1 * np.ones((self.tot_ep_length,), dtype=float)
            observation = self.env.reset(reset_rng=0.05)
            counter, player, done = 0, 0, False
            state_2d = self.env.get_state_2d()
            while not done:
                player %= num_policies
                if render:
                    self.env.render()
                    # time.sleep(self.env.tau)

                state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
                # alternate between the player taking an action and the adversary. After done, switch the policies
                if player == 0:
                    a = policies[policy](state_2d, observation, player)
                else:
                    a = policies[(policy+1) % num_policies](state_2d, observation, player)

                observation, loss, done, info = self.env.step(a, player, next_true_step=True)
                costs[counter] = loss
                counter += 1
                player += 1
            episode_costs.append(costs)
            episode_lengths.append(counter)
        # episode scores are 0<s<1, with s high = good.
        ep_scores = [1+episode_costs[0], -episode_costs[1]]  # 1 for adversary is fallen over
        return ep_scores

    def evaluate_policies(self, testEps, render=False):
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
        for n in range(testEps):

            # calculate the base scores
            curr_player_score, curr_adv_score = self.compare_policies([self.current_action, self.current_action], render)
            curr_player_scores.append(curr_player_score)
            curr_adv_scores.append(curr_adv_score)

            # calculate the scores of the challengers
            chal_player_score, chal_adv_score = self.compare_policies([self.challenger_action, self.current_action], render)
            chal_player_scores.append(chal_player_score)
            chal_adv_scores.append(chal_adv_score)

            Utils.update_progress("CHALLENGER POLICIES TEST EPISODES, ep" + str(n+1)+"(x4)",
                                  (n+1) / testEps, time.time() - start)

        # Compute the stats for the episode, dictionary of mean, var, episode lengths etc
        curr_player = self.compute_statistics(curr_player_scores, 'Player')
        curr_adv = self.compute_statistics(curr_adv_scores, 'Adversary')
        chal_player = self.compute_statistics(chal_player_scores, 'Player')
        chal_adv = self.compute_statistics(chal_adv_scores, 'Adversary')

        # curr_player_mean, curr_adv_mean = sum(curr_player_scores)/testEps, sum(curr_adv_scores)/testEps
        # chal_player_mean, chal_adv_mean = sum(chal_player_scores) / testEps, sum(chal_adv_scores) / testEps
        print("curr_player_mean {}, curr_adv_mean {}".format(curr_player['mean'], curr_adv['mean']))
        print("chal_player_mean {}, chal_adv_mean {}".format(chal_player['mean'], chal_adv['mean']))

        # and approximate the probability of the challenger beating the current policy with a logistic function
        player_improvement = 400*np.log10((1-curr_player['mean'])*chal_player['mean']/((1-chal_player['mean'])*curr_player['mean']))
        adv_improvement = 400*np.log10((1-curr_adv['mean'])*chal_adv['mean']/((1-chal_adv['mean'])*curr_adv['mean']))
        print("player_improvement {}, adv_improvement {}".format(player_improvement, adv_improvement))

        self.chal_player_elo += player_improvement
        self.chal_adv_elo += adv_improvement
        self.env.close()
        print("Elo's:", self.chal_player_elo, self.chal_adv_elo)
        return self.chal_player_elo, self.chal_adv_elo

    def compute_statistics(self, scores, agent='Player'):
        stats = {'score_mean': 0, 'score_var': 0, 'len_mean': 0, 'len_var': 0, 'improvement': 0}
        # extract all data from scores
        print(agent)
        num_episodes = len(scores)
        all_scores = []
        all_lengths = []
        for episode in scores:
            all_scores.extend(episode)
            is_terminal = episode <= 0.0001 if agent == 'Player' else episode >= 0.9999
            all_lengths.append(len(episode[is_terminal == False]))  # extract the episode length

        # compute stats
        stats['len_mean'] = sum(all_lengths)/num_episodes
        # unbiased sample variance
        stats['len_var'] = sum([(ep_len - stats['len_mean'])**2 for ep_len in all_lengths])/(num_episodes-1)

        stats['score_mean']
        # finish doing stats
        # save them all to csv
        # visualise them and then compare them in the report

        return stats


    def save_to_csv(self, file_name, data):
        # maybe add some unpickling for saving whole examples? or to a different function
        file_path = os.path.join('Data', 'TestData', file_name + str(self.policy_iters))
        with open(file_path + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
