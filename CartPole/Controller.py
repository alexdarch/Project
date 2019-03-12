from MCTS import MCTS
from Evaluate import Evaluate
import numpy as np
from collections import deque
from utils import *
import time, csv, os
from random import shuffle
from copy import deepcopy

from utils import *
# from memory_profiler import profile
# from collections import deque
# import matplotlib.pyplot as plt

# from pickle import Pickler, Unpickler


class Controller:
    """
    This class executes the self-play + learning. It uses the functions defined
    in the environment class and NeuralNet. args are specified in main.py.
    """

    def __init__(self, env, nnet, args):
        self.env = env
        self.nnet = nnet  # if loading a nnet
        self.challenger_nnet = self.nnet.__class__(self.env)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args)
        self.policy_examples_history = []  # history of examples from args.policyItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # -------- POLICY STATS ----------
        self.policy_iters = 0
        self.curr_player_elo = 1000
        self.curr_adversary_elo = 1000

        # ------- CONSTS AFFECTING THE VALUE FUNCTION ----------
        self.max_steps_beyond_done = self.env.max_steps_beyond_done
        self.final_discount = 1.0/20    # we want to reduce the discount factor to 1/20 of the first value by the end
        self.discount_factor = self.final_discount ** (1/(self.max_steps_beyond_done - 1))
        self.discount_sum = (1-self.final_discount * self.discount_factor)/(1-self.discount_factor)  # to normalise

    def execute_episode(self):

        """
        This function executes one episode of self-play, with both players acting simultaneously.
        As the game is played, each step is added as a training example to trainExamples. The game is played until
        it reachs max_steps_till_done. After the game finishes, values are calculated from the losses.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (state_2d, pi, pi_adv, v). pi is the MCTS informed policy
                            vector, v is the value defined as an expected future loss, E[L].
        """
        example, observations, losses, policy_predictions = [], [], [], []
        observation = self.env.reset()
        state_2d, loss = self.env.get_state_2d(), self.env.state_loss()
        done = False
        while not done:
            # ---------------------- RECORD STEP ------------------------
            state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
            observations.append(observation)
            losses.append(loss)
            policy_action, policy_adv_action, policy_value = self.nnet.predict(state_2d)
            policy_predictions.append([policy_action.tolist(), policy_adv_action.tolist(), policy_value.tolist()[0]])

            # ---------- GET PROBABILITIES FOR EACH ACTION --------------
            temp = 1  # int(self.env.steps < self.args.tempThreshold)  # act greedily if after 15th step
            pi_player, pi_adv = self.mcts.get_action_prob(state_2d, observation, temp=temp)  # MCTS improved action-prob
            example.append([state_2d, pi_player, pi_adv])

            # ---------- TAKE NEXT STEP PROBABILISTICALLY ---------------
            # take a random choice with pi probability for each action
            actions = [np.random.choice(len(pi_player), p=pi_player), np.random.choice(len(pi_adv), p=pi_adv)]
            observation, loss, done, info = self.env.step(*actions, next_true_step=True)

        # Convert Losses to expected losses (discounted into the future by self.max_steps_beyond_done)
        values = self.get_values_from_losses(losses)
        example = [s_a + [v] for s_a, v in zip(example[:-self.max_steps_beyond_done], values)]
        return example, observations[:-self.max_steps_beyond_done], policy_predictions[:-self.max_steps_beyond_done]

    def get_values_from_losses(self, losses):
        values = []
        for step_idx in range(len(losses) - self.max_steps_beyond_done):
            value, discount = 0, 1
            for i in range(step_idx+1, step_idx+self.max_steps_beyond_done):
                value += discount*losses[i]
                discount *= self.discount_factor
            values.append(value/self.discount_sum)
        return values

    # @Utils.profile
    def policy_iteration(self):
        """
        Performs 'policyIters' iterations with trainEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(self.args.policyIters):
            policy_examples = deque([], maxlen=self.args.maxlenOfQueue)  # double-ended stack
            policy_examples_to_csv = []
            # ------------------- USE CURRENT POLICY TO GENERATE EXAMPLES ---------------------------
            if not self.skipFirstSelfPlay or i > 1:

                start = time.time()
                # Generate a new batch of episodes based on current net
                for eps in range(1, self.args.trainEps+1):
                    self.mcts = MCTS(self.env, self.nnet, self.args)  # reset search tree
                    example_episode, observations, policy_predictions = self.execute_episode()

                    # bookkeeping + plot progress
                    if len(example_episode) > self.args.keepAbove:
                        policy_examples.extend(example_episode)  # add recent episodes to the right
                        policy_examples_to_csv = self.update_csv_examples(policy_examples_to_csv, example_episode,
                                                                          policy_predictions, observations)
                    Utils.update_progress("TRAINING EPISODES ITER: "+str(i)+" ep"+str(eps),
                                          eps/self.args.trainEps,
                                          time.time() - start)
                if self.args.mctsTree:
                    self.mcts.show_tree(values=True)
                    return  # prevents us from overwriting the best model
                np.savez_compressed(r'Data\TrainingData\TrainingExamples'+str(self.policy_iters)+'.npz', *[step[0] for step in policy_examples])  # pickle the state_2d's in a long dict
                self.save_to_csv('TrainingExamples', policy_examples_to_csv)
                self.policy_examples_history.append(policy_examples)  # list of deques

            if len(self.policy_examples_history) > self.args.numItersForTrainExamplesHistory:
                # print("len(trainExamplesHistory) =", len(self.policy_examples_history),
                #     " => remove the oldest trainExamples\n")
                self.policy_examples_history.pop(0)

            # ------ CREATE CHALLENGER POLICY BASED ON EXAMPLES GENERATED BY PREVIOUS POLICY --------
            examples_for_training = []      # take most recent iter examples, shuffle and train
            for policy_itr_examples in self.policy_examples_history:
                examples_for_training.extend(policy_itr_examples)
            shuffle(examples_for_training)

            # essentially deepcopy the nnet to the challenger_nnet
            self.nnet.save_net_architecture(folder=self.args.checkpoint_folder, filename='temp.pth.tar')
            self.challenger_nnet.load_net_architecture(folder=self.args.checkpoint_folder, filename='temp.pth.tar')
            self.challenger_nnet.train_policy(examples=examples_for_training)

            challenger_mcts = MCTS(self.env, self.challenger_nnet, self.args)
            current_mcts = MCTS(self.env, self.nnet, self.args)

            eval = Evaluate(lambda s2d, root: np.argmax(current_mcts.get_action_prob(s2d, root, temp=0), axis=1),
                          lambda s2d, root: np.argmax(challenger_mcts.get_action_prob(s2d, root, temp=0), axis=1),
                            self.env, self.curr_player_elo, self.curr_adversary_elo)
            self.curr_player_elo, self.curr_adversary_elo = eval.evaluate_policies(self.args.testEps)
            print(self.curr_player_elo, self.curr_adversary_elo)

            # ------------------------- COMPARE POLICIES AND UPDATE ---------------------------------
            update = True
            if update:
                self.nnet = deepcopy(self.challenger_nnet)
                # if we are updating then the challenger nnet is the best nnet. Also checkpoint
                self.nnet.save_net_architecture(folder=self.args.checkpoint_folder,
                                                filename='checkpoint_' + str(self.policy_iters) + '.pth.tar')
                self.nnet.save_net_architecture(folder=self.args.checkpoint_folder,
                                                filename='best.pth.tar')
            else:
                # if we didnt update then just reload the nnet we saved before training (though this should do nothing)
                self.challenger_nnet.load_net_architecture(folder=self.args.checkpoint_folder,
                                                filename='temp.pth.tar')
                # note, if there are two PI's in a row and no update then these aren't saved - only previous best ones
            self.policy_iters += 1

    def save_to_csv(self, file_name, data):
        # maybe add some unpickling for saving whole examples? or to a different function
        file_path = os.path.join('Data', 'TrainingData', file_name + str(self.policy_iters))
        with open(file_path + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    @staticmethod
    def update_csv_examples(policy_examples_to_csv, example_episode, policy_predictions, observations):
        policy_examples_to_csv.append([step[3] for step in example_episode])  # only save the value
        policy_examples_to_csv.append([step[2] for step in policy_predictions])  # and the policy-pred value
        policy_examples_to_csv.append([step[1] for step in example_episode])  # save the MCTS actions
        policy_examples_to_csv.append([step[0] for step in policy_predictions])  # save the policy action
        policy_examples_to_csv.append([step[2] for step in example_episode])  # only save the value
        policy_examples_to_csv.append([step[1] for step in policy_predictions])  # and the policy-pred value
        policy_examples_to_csv.append(observations)  # save the observations
        return policy_examples_to_csv
