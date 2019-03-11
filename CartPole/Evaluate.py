from MCTS import MCTS
import numpy as np
from utils import *
import time, csv, os
from utils import *


class Evaluate:
    def __init__(self, curr_policy, challenger_policy, env, starting_elo):
        """
        Evaluates the current policy against the challenger policy. It pits the challenger player vs the current
        adversary and vice versa.
        Input:
            self.curr_action, self.challenger_action: functions that call get_action_prob(state_2d, root_state, temp=0)
                                                      each has its own mcts and therefore policy.
                                             returns: player_probs, adversary_probs
            self.env: The current environment
        """
        # these are actually function pointers to getActionProb, that returns the greedy actions
        self.current_action = curr_policy
        self.challenger_action = challenger_policy
        self.env = env

        self.current_elo = starting_elo
        self.chal_elo = starting_elo

        self.best_policy_stats = {'mean': np.nan, 'avg_length': np.nan}
        self.challenger_policy_stats = {'mean': np.nan, 'avg_length': np.nan}

    def compare_policies(self, render=False):
        policies = [self.current_action, self.challenger_action]
        num_policies = len(policies)
        ep_lengths = []

        for policy in range(num_policies):
            # losses = -1 * np.ones((self.env.max_steps_beyond_done + self.env.steps_till_done + 1,), dtype=float)
            observation = self.env.reset(reset_rng=0.05)
            counter, done = 0, False
            state_2d = self.env.get_state_2d()
            while not done:
                if render:
                    self.env.render()

                state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
                # cycles through the policies so we have a_curr vs adv_chal initially, then reverses
                a, _ = policies[policy](state_2d, observation)
                _, a_adv = policies[(policy+1) % num_policies](state_2d, observation)

                observation, loss, done, info = self.env.step(a, a_adv, next_true_step=True)
                # losses[counter] = loss
                counter += 1
            ep_lengths.append(counter)
        return ep_lengths

    def evaluate_policies(self, testEps, render=False):
        """
        Compares self.challenger_nnet and self.nnet by comparing the means and medians of played episodes
        returns: update = True/False depending on whether some condition is met.
        """

        # ------ PLAY EPISODES WITH CHALLENGER POLICIES ------------
        chal_losses, chal_lengths = [], []
        start = time.time()
        for n in range(testEps):
            chal_adv_length, chal_player_length = self.compare_policies(render)
            # print("Challenger Player Length: {}, Challenger Adversary Length: {}".format(chal_player_length, chal_adv_length))

            # Calculate the new elo scores
            expected_score = 1 / (1+10**((self.current_elo - self.chal_elo)/400))
            score = (chal_player_length/self.env.steps_till_done + 1 - chal_adv_length/self.env.steps_till_done)/2
            self.chal_elo += 32*(score - expected_score)
            # print("Score: {},  Expected Score: {},  New Challenger Elo: {}\n".format(score, expected_score, self.chal_elo))
            # don't change the current one yet

            Utils.update_progress("CHALLENGER POLICIES TEST EPISODES, ep" + str(n+1),
                                  (n+1) / testEps, time.time() - start)

        # ---------- COMPARE NEURAL NETS AND DETERMINE WHETHER TO UPDATE -----------
        # means are -ve, we want the number closest to zero -> chal > curr.
        # Multiplying by 0.95 gets curr -> 0, so better
        # if self.challenger_policy_stats['avg_length'] > self.args.updateThreshold * self.best_policy_stats['avg_length']:
        #     print("UPDATING NEW POLICY FROM avg_length = ", self.best_policy_stats['avg_length'],
        #           "\t TO avg_length = ", self.challenger_policy_stats['avg_length'], "\n")
        #
        #     update = True
        #     self.save_to_csv('BestLosses', chal_losses)
        #     self.best_policy_stats['avg_length'] = self.challenger_policy_stats['avg_length']
        # else:
        #     print("REJECTING NEW POLICY WITH avg_length = ", self.challenger_policy_stats['avg_length'],
        #           "\t AS THE CURRENT avg_length IS = ", self.best_policy_stats['avg_length'], "\n")

        return self.chal_elo
