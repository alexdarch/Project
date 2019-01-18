from MCTS import MCTS
import numpy as np
from Policy import NeuralNet
from utils import *
import time, csv
from random import shuffle
# from collections import deque
# import matplotlib.pyplot as plt

# from pickle import Pickler, Unpickler


class Controller:
    """
    This class executes the self-play + learning. It uses the functions defined
    in the environment class and NeuralNet. args are specified in main.py.
    """
    improvements = 0
    iters = 0

    def __init__(self, env, nnet, args):
        self.env = env
        self.nnet = nnet
        self.challenger_nnet = self.nnet.__class__(env)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.policyItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # ------- CONSTS AFFECTING THE VALUE FUNCTION ----------
        self.max_steps_beyond_done = self.env.max_steps_beyond_done
        self.final_discount = 1.0/20    # we want to reduce the discount factor to 1/20 of the first value by the end
        self.discount_factor = self.final_discount ** (1/(self.max_steps_beyond_done - 1))
        self.discount_sum = (1-self.final_discount * self.discount_factor)/(1-self.discount_factor)  # to normalise

    def execute_episode(self):

        """
        This function executes one episode of self-play, starting with player 1. As the game is played, each turn
        is added as a training example to trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (state_2d, pi, v). pi is the MCTS informed policy
                            vector, v is +1 if the player eventually won the game, else -1.
        """
        example, losses = [], []
        observation = self.env.reset()
        episode_step = 0
        state_2d = self.env.get_state_2d(observation)

        while True:
            episode_step += 1
            state_2d = self.env.get_state_2d(observation, state_2d)

            # ---------- GET PROBABILITIES FOR EACH ACTION --------------
            temp = int(episode_step < self.args.tempThreshold)  # act greedily if after 15th step
            pi = self.mcts.get_action_prob(state_2d, self.env, temp=temp)  # MCTS improved action-prob

            # ---------- TAKE NEXT STEP PROBABILISTICALLY ---------------
            action = np.random.choice(len(pi), p=pi)  # take a random choice with pi probability for each action
            observation, loss, done, info = self.env.step(action)

            # ---------------------- RECORD STEP ------------------------
            example.append([state_2d, pi])
            losses.append(loss)

            if done:
                break

        # ------------------ DO N STEPS AFTER DONE ----------------------
        for extra_steps in range(self.max_steps_beyond_done):
            state_2d = self.env.get_state_2d(observation, state_2d)
            pi = self.mcts.get_action_prob(state_2d, self.env, temp=temp)  # MCTS improved policy

            action = np.random.choice(len(pi), p=pi)  # take a random choice with pi probability for each action
            observation, loss, done, info = self.env.step(action)
            losses.append(loss)  # only need to record losses post done

        # Convert Losses to expected losses (discounted into the future by self.max_steps_beyond_done)
        values = self.value_function(losses)
        example = [s_a + [v] for s_a, v in zip(example, values)]
        return example

    def greedy_episode(self, g_mcts):
        # No self play yet!!
        losses = []
        observation, done = self.env.reset(), False
        state_2d = self.env.get_state_2d(observation)

        while not done:
            state_2d = self.env.get_state_2d(observation, state_2d)

            # ------------- GET PROBABILITIES FOR EACH ACTION --------------
            pi = g_mcts.get_action_prob(state_2d, self.env, temp=0)  # MCTS improved policy
            # pi, v = neural_net.predict(state_2d)  # Standard nnet predicted policy (for now)

            # ---------- TAKE NEXT STEP GREEDILY AND SAVE DATA--------------
            action = np.argmax(pi)
            observation, loss, done, info = self.env.step(action)
            losses.append(loss)

        return losses

    def value_function(self, losses: list) -> list:
        values = []
        for step_idx in range(len(losses) - self.max_steps_beyond_done):
            value, discount = 0, 1
            for i in range(step_idx+1, step_idx+self.max_steps_beyond_done):
                value += discount*losses[i]
                discount *= self.discount_factor
            values.append(value/self.discount_sum)
        return values

    def policy_iteration(self):
        """
        Performs 'policyIters' iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(self.args.policyIters):

            policy_examples = []  # deque([], maxlen=self.args.maxlenOfQueue)  # double-ended stack
            # ------------------- USE CURRENT POLICY TO GENERATE EXAMPLES ---------------------------
            if not self.skipFirstSelfPlay or i > 1:

                start = time.time()
                # Generate a new batch of episodes based on current net
                for eps in range(1, self.args.numEps+1):
                    self.mcts = MCTS(self.env, self.nnet, self.args)  # reset search tree
                    policy_examples += self.execute_episode()

                    # bookkeeping + plot progress
                    Utils.update_progress("EXECUTING EPISODES UNDER POLICY ITER "+str(i+1), eps/self.args.numEps,
                                          time.time() - start)

                # self.trainExamplesHistory.append(policy_examples)  # save all examples

            # ------ CREATE CHALLENGER POLICY BASED ON EXAMPLES GENERATED BY PREVIOUS POLICY --------
            shuffle(policy_examples)   # should shuffle examples before training
            self.challenger_nnet = NeuralNet(self.env)  # create a new net to train
            self.challenger_nnet.train_policy(examples=policy_examples)

            # ------------------------- COMPARE POLICIES AND UPDATE ---------------------------------
            Controller.iters += 1
            update = self.policy_improvement()
            if update:
                self.nnet = self.challenger_nnet
                # if we are updating then the challenger nnet is the best nnet. Also checkpoint
                self.nnet.save_net_architecture(folder=self.args.checkpoint,
                                          filename='checkpoint_' + str(Controller.iters) + '.pth.tar')
                self.nnet.save_net_architecture(folder=self.args.checkpoint,
                                          filename='best.pth.tar')
            else:
                # if the challenger is worse than the current the the current is the best nnet yet
                # so save the current nnet as the best nnet
                self.nnet.save_net_architecture(folder=self.args.checkpoint,
                                          filename='best.pth.tar')
                # note, if there are two PI's in a row and no update then these aren't saved - only previous best ones

    def policy_improvement(self):
        """
        Compares self.challenger_nnet and self.nnet by comparing the means and medians
        returns: update = True/False depending on whether some condition is met.
        """
        update = False
        curr_losses, chal_losses = [], []
        curr_csv, chal_csv = [], []
        start = time.time()
        for n in range(1, self.args.testIters+1):
            # ---- Play an episode with the current policy ----
            curr_mcts = MCTS(self.env, self.nnet, self.args)  # reset mcts tree for each episode
            episode_curr_losses = self.greedy_episode(curr_mcts)
            curr_losses.extend(episode_curr_losses)
            curr_csv.append(episode_curr_losses)

            # ---- Play an episode with challenger policy ----
            chal_mcts = MCTS(self.env, self.challenger_nnet, self.args)
            episode_chal_losses = self.greedy_episode(chal_mcts)
            chal_losses.extend(episode_chal_losses)
            chal_csv.append(episode_chal_losses)

            # progress
            Utils.update_progress("GREEDY POLICIES EXECUTION, ep"+str(n),
                                  n / self.args.testIters, time.time() - start)

        Controller.improvements += 1
        with open(r'Data\Challenger Losses' + str(Controller.improvements) + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(chal_csv)
        with open(r'Data\Current Losses' + str(Controller.improvements) + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(curr_csv)

        # print(chal_losses)
        # print(curr_losses)
        # # -------- CALCULATE GAUSSIAN DISTRIBUTIONS FOR CURRENT AND CHALLENGER LOSSES -------------
        # # means and variances
        # curr_mean, curr_variance = np.mean(curr_losses), np.var(curr_losses)
        # chal_mean, chal_variance = np.mean(chal_losses), np.var(chal_losses)
        #
        # third_std = 4*max(np.sqrt(chal_variance), np.sqrt(curr_variance))   # calculate 4 standard deviations
        # avg_mean = np.mean([curr_mean, chal_mean])  # and a roughly average centre
        # # just want a range that roughly covers 4 standard deviations around the means (which are about equal usually)
        # x_range = np.linspace(avg_mean - third_std, avg_mean + third_std, 200)
        #
        # # calculate a y=x^2 transformed normal pdf over the losses of each
        # curr_y = np.exp(-0.5 * (x_range-curr_mean)**2 / curr_variance) / np.sqrt(2*curr_variance*np.pi)
        # chal_y = np.exp(-0.5 * (x_range-chal_mean)**2 / chal_variance) / np.sqrt(2*chal_variance*np.pi)
        #
        # # -------------- CALCULATE p(X-Y > 0) DIFFERENCE OF GAUSSIANS --------------
        # # # Calculate difference of gaussians stats (mode == mean for gaussians)
        # diff_mean, diff_variance = curr_mean - chal_mean, np.sqrt(curr_variance + chal_variance)
        # x_difference = np.linspace(diff_mean-3*np.sqrt(diff_variance), diff_mean+3*np.sqrt(diff_variance), 200)
        # y_difference = np.exp(-0.5*(x_difference-diff_mean)**2/diff_variance)/np.sqrt(abs(x_difference)*2*diff_variance*np.pi)
        # zero_idx = (np.abs(x_difference - 0)).argmin()  # find the index where x=0 (or closest to)
        # prob_chal_greater_than_curr = np.trapz(y_difference[zero_idx:], x_difference[zero_idx:])
        #
        # plt.plot(x_range, curr_y)
        # plt.plot(x_range, chal_y)
        # plt.legend(["current policy", "challenger policy"])
        # plt.gca().set_prop_cycle(None)  # reset the colour cycles
        # plt.hist(curr_losses, bins=20, density=True, alpha=0.3)
        # plt.hist(chal_losses, bins=20, density=True, alpha=0.3)
        #
        # plt.plot(x_difference, y_difference)
        # plt.show()
        # print("prob_chal_greater_than_curr: ", prob_chal_greater_than_curr)

        # MOVE THIS INTO A self.curr_nnet_stats dict and then compare directly in policy iteration?
        # defo dont need to do episodes with the current nnet again, can just reuse information...
        # ---------- COMPARE NEURAL NETS AND DETERMINE WHETHER TO UPDATE -----------
        # means are -ve, we want the number closest to zero -> chal > curr.
        # Multiplying by 0.95 gets curr -> 0, so better
        if np.mean(chal_losses) > self.args.updateThreshold*np.mean(curr_losses):
            update = True
            print("UPDATING NEW POLICY FROM MEAN = ", np.mean(curr_losses), "\t TO MEAN = ", np.mean(chal_losses))
        else:
            print("REJECTING NEW POLICY WITH MEAN = ", np.mean(chal_losses),
                  "\t AS THE CURRENT MEAN IS = ", np.mean(curr_losses))

        return update
