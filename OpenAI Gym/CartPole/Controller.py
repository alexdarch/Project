from Policy import Net
import numpy as np
from collections import Counter
import torch
from copy import deepcopy
import MCTS


class Controller():

    def __init__(self, game, args, nnet=None):

        self.nnet = nnet  # the nnet "is part of" the controller -> composition (or aggregation?.. implemented by pointer/reference in c++)
        # self.mcts = MCTS()
        self.args = args
        self.env = game

    def policyIteration(self):

        scores = np.array([])
        # self.nnet = Net() # don't actually need to initiate the "prev_nnet",
                            # since it is defined when we create a controller object
        init_examples, curr_mean, curr_median = self.initialExamples()  # Don't need to pass a model
        a_loss, v_loss, batch_acc = self.nnet.train_model(examples=init_examples)

        for i in range(self.args.policyUpdates):

            # ----- GENERATE A NEW BATCH OF EPISODES BASED ON THE CURRENT NET----------
            exampleBatch = []
            for e in range(self.args.policyEpisodes):
                example = self.executeEpisode()
                scores = np.append(scores, example[0, 5])

                if len(exampleBatch) == 0:
                    exampleBatch = example
                else:
                    exampleBatch = np.vstack((exampleBatch, example))

            # -------- CREATE CHALLENGER POLICY BASED ON EXAMPLES GENERATED BY PREVIOUS POLICY -------------------
            new_nnet = Net(self.env, self.args)  # create a new net to train
            a_loss, v_loss, batch_acc = new_nnet.train_model(examples=exampleBatch)

            # -------- PRINT STATS ON NEW POLICY -------------
            new_mean, new_median = np.mean(scores), np.median(scores)
            print('Average accepted score: ', new_mean)
            print('Median score for accepted scores: ', new_median)
            print(Counter(scores))
            print("Current Policy: ", curr_mean, curr_median)

            # ---------- COMPARE AND UPDATE POLICIES --------------
            if new_mean >= curr_mean and new_median >= curr_median:
                self.nnet = new_nnet
                curr_mean, curr_median = new_mean, new_median
                print("Policy Updated!")
                print("New Policy: ", curr_mean, curr_median)

        return self.nnet

    def executeEpisode(self, init_egs=False):
        ''' Generate and example episode of [4 x observation(t), action(t), E[return(t)]].
            All values are in a (n x 6) numpy array where n is the number of steps for the
            episode to finish or the limit of 200 steps'''
        score = 0
        example = np.zeros((self.args.goal_steps, 6))
        prev_observation = self.env.reset()  # list of 4 elements

        # --------- ITERATE UP TO 500 STEPS PER EPISODE -------------
        for t in range(self.args.goal_steps):

            # --------- GENERATE ACTION ------------
            # We can generate random actions or actions from the previous policy (i.e. prev nnet)
            if init_egs or t == 0:
                action = self.env.action_space.sample()  # choose random action (0-left or 1-right)
            else:
                x = torch.tensor(prev_observation, dtype=torch.float)
                action_prob, e_score = self.nnet.forward(x)
                action = np.argmax(action_prob.detach().numpy())

            observation, reward, done, info = self.env.step(action)

            # --------- STORE STATE-ACTION PAIR + SCORE ------------
            example[t, 0:4] = prev_observation[0:4]
            example[t, 4:6] = [action, score]

            prev_observation = np.array(observation)
            score += reward  # +1 for every frame we haven't fallen

            if done:
                break

        example[:, 5] = score - example[:, 5]  # Convert scores to E[return]
        return example[0:int(score), :]  # we only want to return the parts with actual values

    def initialExamples(self):
        allExamples = []
        accepted_scores = np.array([])  # just the scores that met our threshold

        # --------------- ITERATE THROUGH 10000 EPISODE ------------------
        for _ in range(self.args.initial_games):

            exampleGame = self.executeEpisode(init_egs=True)

            # --------- SAVE EXAMPLE (EPISODE) IF (SCORE > THRESHOLD) ----------
            # Note, it does not save the score! Therefore all episodes with score > threshold
            # are treated equally (not the best way of doing this!)
            if exampleGame[0, 5] >= self.args.score_requirement:

                accepted_scores = np.append(accepted_scores, exampleGame[0, 5])

                if len(allExamples) == 0:
                    allExamples = exampleGame
                else:
                    allExamples = np.vstack((allExamples, exampleGame))

        # -------- PRINT STATS ------------
        avg_mean, avg_median = np.mean(accepted_scores), np.median(accepted_scores)
        print('Average accepted score: ', avg_mean)
        print('Median score for accepted scores: ', avg_median)
        print(Counter(accepted_scores))
        print(len(accepted_scores))

        return allExamples, avg_mean, avg_median
