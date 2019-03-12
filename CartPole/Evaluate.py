from MCTS import MCTS
import numpy as np
from utils import *
import time, csv, os
from utils import *


class Evaluate:
    def __init__(self, curr_policy, challenger_policy, env, starting_player_elo, starting_adv_elo):
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

        self.curr_player_elo = starting_player_elo
        self.curr_adv_elo = starting_adv_elo
        self.chal_player_elo = starting_player_elo
        self.chal_adv_elo = starting_adv_elo

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
        start = time.time()
        for n in range(testEps):
            chal_adv_length, chal_player_length = self.compare_policies(render)
            print("Challenger Player Length: {}, Challenger Adversary Length: {}".format(chal_player_length, chal_adv_length))

            # Calculate the new elo scores
            expected_player_score = 1 / (1+10**((self.curr_player_elo - self.chal_player_elo)/400))
            expected_adv_score = 1 / (1+10**((self.curr_adv_elo - self.chal_adv_elo)/400))
            player_score = chal_player_length/self.env.steps_till_done
            adv_score = 1 - chal_adv_length/self.env.steps_till_done
            self.chal_player_elo += 32*(player_score - expected_player_score)
            self.chal_adv_elo += 32*(adv_score - expected_adv_score)
            # print("Score: {},  Expected Score: {},  New Challenger Elo: {}\n".format(score, expected_score, self.chal_elo))
            # don't change the current one yet

            Utils.update_progress("CHALLENGER POLICIES TEST EPISODES, ep" + str(n+1),
                                  (n+1) / testEps, time.time() - start)

        return self.chal_player_elo, self.chal_adv_elo

    def save_to_csv(self, file_name, data):
        # maybe add some unpickling for saving whole examples? or to a different function
        file_path = os.path.join('Data', 'TestData', file_name + str(self.policy_iters))
        with open(file_path + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
