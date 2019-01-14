from collections import deque
from MCTS import MCTS
import numpy as np
from random import shuffle
from Policy import NeuralNet
from utils import *
import time
# from pickle import Pickler, Unpickler


class Controller:
    """
    This class executes the self-play + learning. It uses the functions defined
    in the environment class and NeuralNet. args are specified in main.py.
    """

    def __init__(self, env, nnet, args):
        self.env = env
        self.steps_beyond_done = self.env.steps_beyond_done
        self.nnet = nnet
        self.challenger_nnet = self.nnet.__class__(env)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.policyItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1. As the game is played, each turn
        is added as a training example to trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi, v). pi is the MCTS informed policy
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
            temp = int(episode_step < self.args.tempThreshold)  # temperature = 0 if first step, 1 otherwise
            # pi = self.mcts.getActionProb(state2D, self.env, temp=temp) # MCTS improved policy
            pi, v = self.nnet.predict(state_2d)  # Standard nnet predicted policy (for now)

            # ---------- TAKE NEXT STEP PROBABILISTICALLY ---------------
            action = np.random.choice(len(pi), p=pi)  # take a random choice with pi probability for each action
            observation, loss, done, info = self.env.step(action)

            pi = self.env.get_action(action)  # DELETE WHEN USING MCTS (converts action to [1, 0] or [0, 1] form)
            # print("ACTION PROB: ", pi, "\t\tACTION TAKEN: ", action)

            # ---------------------- RECORD STEP ------------------------
            example.append([state_2d, pi])
            losses.append(loss)

            if done:
                break

        # ------------------ DO N STEPS AFTER DONE ----------------------
        for extra_steps in range(self.steps_beyond_done):
            state_2d = self.env.get_state_2d(observation, state_2d)
            # pi = self.mcts.getActionProb(state2D, self.env, temp=temp) # MCTS improved policy
            pi, v = self.nnet.predict(state_2d)  # Standard nnet predicted policy (for now)

            action = np.random.choice(len(pi), p=pi)  # take a random choice with pi probability for each action
            observation, loss, done, info = self.env.step(action)
            losses.append(loss)  # only need to record losses post done

        # Convert Losses to expected losses (discounted into the future by self.steps_beyond_done)
        values = self.value_function(losses)
        example = [s_a + [v] for s_a, v in zip(example, values)]
        return example

    def greedy_episode(self, g_mcts, neural_net):
        # No self play yet!!
        losses = []
        observation, done = self.env.reset(), False
        state_2d = self.env.get_state_2d(observation)

        while not done:
            state_2d = self.env.get_state_2d(observation, state_2d)

            # ---------- GET PROBABILITIES FOR EACH ACTION --------------
            # pi = g_mcts.getActionProb(state2D, self.env, temp=0) # MCTS improved policy
            pi, v = neural_net.predict(state_2d)  # Standard nnet predicted policy (for now)

            # ---------- TAKE NEXT STEP PROBABILISTICALLY ---------------
            action = np.argmax(pi)
            observation, loss, done, info = self.env.step(action)
            losses.append(loss)

        return np.mean(losses), np.median(losses)

    def value_function(self, losses):
        values = []
        for step_idx in range(len(losses) - self.steps_beyond_done):
            beta, change = 1.0, 1.0/self.steps_beyond_done
            value = 0
            for i in range(step_idx+1, step_idx+self.steps_beyond_done):
                value += beta*losses[i]
                beta -= change
            values.append(value)
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
                    eps_time = time.time() - start
                    Utils.update_progress("EXECUTING EPISODES UNDER POLICY ITER "+str(i+1), eps/self.args.numEps, eps_time)

                # self.trainExamplesHistory.append(policy_examples)  # save all examples

            # ------ CREATE CHALLENGER POLICY BASED ON EXAMPLES GENERATED BY PREVIOUS POLICY --------
            self.challenger_nnet = NeuralNet(self.env)  # create a new net to train
            self.challenger_nnet.train_policy(examples=policy_examples)  # should shuffle examples before training

            # ------------------------- COMPARE POLICIES AND UPDATE ---------------------------------
            update, new_tot_mean, new_tot_median = self.policy_improvement()
            if update:
                print("ACCEPTING NEW MODEL WITH MEAN: ", new_tot_mean, "\tMEDIAN: ", new_tot_median, "\n")
                self.nnet = self.challenger_nnet
                # save checkpoint?
            else:
                print("REJECTING NEW MODEL\n")
                # load checkpoint?

    def policy_improvement(self):
        """
        Compares self.challenger_nnet and self.nnet by comparing the means and medians
        returns: update = True/False depending on whether some condition is met.
        """
        update = False
        curr_means, curr_medians = [], []
        chal_means, chal_medians = [], []
        start = time.time()
        for n in range(1, self.args.testIters+1):
            # ---- Play an episode with the current policy ----
            curr_mcts = MCTS(self.env, self.nnet, self.args)  # reset mcts tree for each episode
            curr_mean, curr_median = self.greedy_episode(curr_mcts, self.nnet)
            curr_means.append(curr_mean)
            curr_medians.append(curr_median)

            # ---- Play an episode with challenger policy ----
            chal_mcts = MCTS(self.env, self.nnet, self.args)
            chal_mean, chal_median = self.greedy_episode(chal_mcts, self.challenger_nnet)
            chal_means.append(chal_mean)
            chal_medians.append(chal_median)

            # progress
            Utils.update_progress("GREEDY POLICIES EXECUTION, ep"+str(n),
                                  n / self.args.testIters, time.time() - start)

        # ---------- COMPARE NEURAL NETS AND DETERMINE WHETHER TO UPDATE -----------
        ret_sum_mean, ret_sum_medians = sum(curr_means), sum(curr_medians)
        if sum(chal_means) > ret_sum_mean and sum(chal_medians) > ret_sum_medians:
            ret_sum_mean, ret_sum_medians = sum(chal_means), sum(chal_medians)
            update = True

        return update, ret_sum_mean, ret_sum_medians
