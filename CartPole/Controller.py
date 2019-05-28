from MCTS import MCTS
import numpy as np
from collections import deque
import time, csv, os
from random import shuffle
from utils import *
# from memory_profiler import profile

class Controller:
    """
    This class executes the self-play + learning. It uses the functions defined
    in the environment class and NeuralNet. args are specified in main.py.
    """

    def __init__(self, env, nnet, args):
        self.env = env
        self.nnet = nnet  # if loading a nnet
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args)
        self.policy_examples_history = []  # history of examples from args.policyItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # ------- CONSTS AFFECTING THE VALUE FUNCTION ----------
        self.max_steps_beyond_done = self.env.max_steps_beyond_done
        self.final_discount = 1.0/20    # we want to reduce the discount factor to 1/20 of the first value by the end
        self.discount_factor = self.final_discount ** (1/(self.max_steps_beyond_done - 1))
        self.discount_sum = (1-self.final_discount * self.discount_factor)/(1-self.discount_factor)  # to normalise

    def execute_episode(self):

        """
        This function executes one episode of self-play, with both agents acting simultaneously.
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
        agent = 0  # start with the player pushing on the cart (not the pole)
        while not done:
            # ---------------------- RECORD STEP ------------------------
            agent %= 2  # keep agent rotating through 0 and 1
            if self.args.renderEps:
                self.env.render()

            state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
            observations.append(observation)
            losses.append(loss)
            policy_action, policy_value = self.nnet.predict(state_2d, agent)
            policy_predictions.append([policy_action, policy_value])

            # ---------- GET PROBABILITIES FOR EACH ACTION --------------
            temp = int(self.env.steps < self.args.tempThreshold)  # act greedily if after 15th step
            pi = self.mcts.get_action_prob(state_2d, observation, agent, temp=temp)  # MCTS improved action-prob
            example.append([state_2d, pi, agent])

            # ---------- TAKE NEXT STEP PROBABILISTICALLY ---------------
            # take a random choice with pi probability for each action
            action = np.random.choice(len(pi), p=pi)
            observation, loss, done, info = self.env.step(action, agent, next_true_step=True)
            agent += 1

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
        # save the initial policy
        self.nnet.save_net_architecture(folder=self.args.checkpoint_folder, filename='checkpoint_' + str(0) + '.pth.tar')
        for policy_iters in range(self.args.policyIters):
            self.env.increment_handicap(policy_iters)
            policy_examples = deque([], maxlen=self.args.maxlenOfQueue)  # double-ended stack
            policy_examples_to_csv = []
            # ------------------- USE CURRENT POLICY TO GENERATE EXAMPLES ---------------------------
            start = time.time()
            trainEps = self.args.initialTrainEps if policy_iters < self.args.unopposedTrains else self.args.trainEps
            for eps in range(1, trainEps+1):
                self.mcts = MCTS(self.env, self.nnet, self.args)  # reset search tree
                example_episode, observations, policy_predictions = self.execute_episode()

                # bookkeeping + plot progress
                policy_examples.extend(example_episode)  # add recent episodes to the right
                policy_examples_to_csv = self.update_csv_examples(policy_examples_to_csv, example_episode,
                                                                  policy_predictions, observations)
                Utils.update_progress("TRAINING EPISODES ITER: "+str(policy_iters)+" ep"+str(eps),
                                      eps/trainEps,
                                      time.time() - start)

            # ----------------- SAVE EXAMPLES TO QUEUE AND CSV (+ tree and render close) -------------------
            self.env.close()
            if self.args.mctsTree:
                self.mcts.show_tree()
                return  # prevents us from overwriting the best model

            np.savez_compressed(r'Data\TrainingData\TrainingExamples'+str(policy_iters)+'.npz', *[step[0] for step in policy_examples])  # pickle the state_2d's in a long dict
            self.save_to_csv('TrainingExamples', policy_examples_to_csv, policy_iters)
            self.policy_examples_history.append(policy_examples)  # list of deques

            if len(self.policy_examples_history) > self.args.numItersForTrainExamplesHistory:
                self.policy_examples_history.pop(0)

            # ------------- ONLY TRAIN ON THE LAST N ITERATIONS --------------
            examples_for_training = []      # take most recent iter examples, shuffle and train
            for policy_itr_examples in self.policy_examples_history:
                examples_for_training.extend(policy_itr_examples)
            shuffle(examples_for_training)  # don't want to shuffle the ordered examples before training

            # ---------- RETRAIN NEURAL NETWORKS AND CHECKPOINT --------------
            # note, this is AlphaZero, therefore always retrain (no evaluation)
            self.nnet.train_policy(examples=examples_for_training)
            self.nnet.save_net_architecture(folder=self.args.checkpoint_folder,
                                            filename='checkpoint_' + str(policy_iters) + '.pth.tar')

            if policy_iters == self.args.unopposedTrains-1:
                    print('Clearing Examples History')
                    self.policy_examples_history = []   # don't train the adversary on episodes where it wasn't active


    def save_to_csv(self, file_name, data, policy_iters):
        # maybe add some unpickling for saving whole examples? or to a different function
        folder = os.path.join('Data', 'TrainingData')
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)

        file_path = os.path.join(folder, file_name + str(policy_iters))
        with open(file_path + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    @staticmethod
    def update_csv_examples(policy_examples_to_csv, example_episode, policy_predictions, observations):
        policy_examples_to_csv.append([step[3] for step in example_episode])  # only save the value
        policy_examples_to_csv.append([step[1] for step in policy_predictions])  # and the policy-pred value
        policy_examples_to_csv.append([step[1] for step in example_episode])  # save the MCTS actions
        policy_examples_to_csv.append([step[0] for step in policy_predictions])  # save the policy action
        policy_examples_to_csv.append([step[2] for step in example_episode])  # save the agent
        policy_examples_to_csv.append(observations)  # save the observations
        return policy_examples_to_csv
