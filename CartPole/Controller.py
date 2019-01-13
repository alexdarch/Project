from collections import deque
from MCTS import MCTS
import numpy as np
from random import shuffle
from Policy import NeuralNet
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
# from pickle import Pickler, Unpickler


class Controller:
    """
    This class executes the self-play + learning. It uses the functions defined
    in the environment class and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.env = game
        self.steps_beyond_done = self.env.steps_beyond_done
        self.nnet = nnet
        # self.pnet = self.nnet.__class__(self.env)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.policyItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def execute_episode(self, greedy=False):
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
        print("--------------------- NEW EPISODE -------------------------")

        while True:
            episode_step += 1
            state_2d = self.env.get_state_2d(observation, state_2d)

            # ---------- GET PROBABILITIES FOR EACH ACTION --------------
            temp = int(episode_step < self.args.tempThreshold)  # temperature = 0 if first step, 1 otherwise
            # pi = self.mcts.getActionProb(state2D, self.env, temp=temp) # MCTS improved policy
            pi, v = self.nnet.predict(state_2d)  # Standard nnet predicted policy (for now)

            # ---------- TAKE NEXT STEP PROBABILISTICALLY ---------------
            if greedy:
                action = np.argmax(pi)
            else:
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

            action = np.random.choice(len(pi), p=pi)  # take a random choice with pi probability associated with each
            observation, loss, done, info = self.env.step(action)
            losses.append(loss)  # only need to record losses post done

        # Convert Losses to expected losses (discounted into the future by self.steps_beyond_done)
        values = self.value_function(losses)
        example = [s_a + [v] for s_a, v in zip(example, values)]
        return example

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
        for i in range(1, self.args.policyIters + 1):

            policy_examples = deque([], maxlen=self.args.maxlenOfQueue)  # double-ended stack
            # ------------------- USE CURRENT POLICY TO GENERATE EXAMPLES ---------------------------
            print('-----------------ITER ' + str(i) + '--------------------')
            if not self.skipFirstSelfPlay or i > 1:

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                # Generate a new batch of episodes based on current net
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.env, self.nnet, self.args)  # reset search tree
                    policy_examples += self.execute_episode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()

                bar.finish()
                # self.trainExamplesHistory.append(policy_examples)  # save all examples

            # ------ CREATE CHALLENGER POLICY BASED ON EXAMPLES GENERATED BY PREVIOUS POLICY --------
            # shuffle examples before training
            challenger_nnet = NeuralNet(self.env)  # create a new net to train
            challenger_nnet.train_policy(examples=shuffle(policy_examples))

            # ------------------------- COMPARE POLICIES AND UPDATE ---------------------------------
            # ought to pass in the nnets because we are compare policy_examples using greedy policies
            update = self.policy_improvement(challenger_nnet)
            if update:
                print("ACCEPTING NEW MODEL")
                self.nnet = challenger_nnet
                # save checkpoint?
            else:
                print("REJECTING NEW MODEL")
                # load checkpoint?


            # # training new network, keeping a copy of the old one
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)
            #
            # self.nnet.train(trainExamples)
            # nmcts = MCTS(self.game, self.nnet, self.args)
            #
            # print('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            #
            # print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            #     print('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     print('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def policy_improvement(self, challenger_policy):

        # could execute N episodes with self.nnet and a greedy policy,
        # then change self.nnet = challenger_policy and repeat
        # then change it back to the original policy? Or just change execute_episode to
        # not use self.nnet? Or use the pnet thing in orthello?

        # # -------- PRINT STATS ON NEW POLICY -------------
        # values = [x[2] for x in challenger_examples]
        # print(values)
        # new_mean, new_median = np.mean(values), np.median(values)
        # print('Average accepted score: ', new_mean)
        # print('Median score for accepted scores: ', new_median)
        # print("Current Policy: ", curr_mean, curr_median)
        #
        # # ---------- COMPARE AND UPDATE POLICIES --------------
        # if new_mean >= curr_mean and new_median >= curr_median:
        #     self.nnet = new_nnet
        #     curr_mean, curr_median = new_mean, new_median
        #     print("Policy Updated!")
        #     print("New Policy: ", curr_mean, curr_median)

        return True
