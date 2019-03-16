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
            self.curr_action, self.challenger_action: functions that call get_action_prob(state_2d, root_state, player, temp=0)
                                                      each has its own mcts and therefore policy.
                                             returns: player_probs, adversary_probs
            self.env: The current environment
        """
        # these are actually function pointers to getActionProb, that returns the greedy actions
        self.current_action = curr_policy
        self.challenger_action = challenger_policy
        self.env = env
        self.norm = self.env.steps_till_done + self.env.max_steps_beyond_done + 1

        self.chal_player_elo = starting_player_elo
        self.chal_adv_elo = starting_adv_elo

        self.best_policy_stats = {'mean': np.nan, 'avg_length': np.nan}
        self.challenger_policy_stats = {'mean': np.nan, 'avg_length': np.nan}

    def compare_policies(self, policies, render=False):
        num_policies = len(policies)
        ep_lengths = []
        for policy in range(num_policies):
            # losses = -1 * np.ones((self.env.max_steps_beyond_done + self.env.steps_till_done + 1,), dtype=float)
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
                # losses[counter] = loss
                counter += 1
                player += 1
            ep_lengths.append(counter)

        ep_scores = [ep_lengths[0]/self.norm, 1-ep_lengths[1]/self.norm]
        return ep_scores

    def evaluate_policies(self, testEps, render=True):
        """
        Compares self.challenger_nnet and self.nnet by comparing the means and medians of played episodes
        returns: update = True/False depending on whether some condition is met.
        """
        # record the expected & actual player scores, and score squared to get the variance
        Es_player, Es2_player = [], []
        Es_adv, Es2_adv = [], []
        s_player, s2_player = [], []
        s_adv, s2_adv = [], []
        # ------ PLAY EPISODES WITH CHALLENGER POLICIES ------------
        start = time.time()
        for n in range(testEps):

            # calculate the base scores
            Eplayer_score, Eadv_score = self.compare_policies([self.current_action, self.current_action], render=False)
            Es_player.append(Eplayer_score); Es2_player.append(Eplayer_score ** 2)
            Es_adv.append(Eadv_score); Es2_adv.append(Eadv_score ** 2)

            # calculate the scores of the challengers
            player_score, adv_score = self.compare_policies([self.challenger_action, self.current_action], render)
            s_player.append(player_score); s2_player.append(player_score ** 2)
            s_adv.append(adv_score); s2_adv.append(adv_score ** 2)

            Utils.update_progress("CHALLENGER POLICIES TEST EPISODES, ep" + str(n+1)+"(x4)",
                                  (n+1) / testEps, time.time() - start)

        # Compute the means and variances
        Emean_player, Emean_adv = sum(Es_player)/testEps, sum(Es_adv)/testEps
        Evar_player, Evar_adv = sum(Es2_player)/testEps - Emean_player**2, sum(Es2_adv)/testEps - Emean_adv**2
        mean_player, mean_adv = sum(s_player) / testEps, sum(s_adv) / testEps
        var_player, var_adv = sum(s2_player) / testEps - mean_player ** 2, sum(s2_adv) / testEps - mean_adv ** 2
        print("Emean_player {}, Emean_adv {}, Evar_player {}, Evar_adv {}".format(Emean_player, Emean_adv, Evar_player, Evar_adv))
        print("mean_player {}, mean_adv {}, var_player {}, var_adv {}".format(mean_player, mean_adv, var_player,
                                                                                  var_adv))

        # and approximate the probability of the challenger beating the current policy with a logistic function
        prob_player_improvement = 1/(1 + np.exp((Emean_player-mean_player) / np.sqrt(np.pi*(Evar_player+var_player)/8)))
        prob_adv_improvement = 1/(1 + np.exp((Emean_adv-mean_adv) / np.sqrt(np.pi*(Evar_adv + var_adv)/8)))
        print("prob_player_improvement {}, prob_adv_improvement {}".format(prob_player_improvement, prob_adv_improvement))

        self.chal_player_elo += 400*np.log10(prob_player_improvement/(1-prob_player_improvement))
        self.chal_adv_elo += 400*np.log10(prob_adv_improvement/(1-prob_adv_improvement))
        self.env.close()
        print("Elo's:", self.chal_player_elo, self.chal_adv_elo)
        return self.chal_player_elo, self.chal_adv_elo

    def save_to_csv(self, file_name, data):
        # maybe add some unpickling for saving whole examples? or to a different function
        file_path = os.path.join('Data', 'TestData', file_name + str(self.policy_iters))
        with open(file_path + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
