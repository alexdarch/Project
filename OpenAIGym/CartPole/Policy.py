import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter


class Net(nn.Module):

    def __init__(self, env, args):
        self.env = env
        self.args = args
        super(Net, self).__init__()

        self.l1 = nn.Linear(10000, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 32)

        self.dp = nn.Dropout(p=self.args.dropout)  # Suragnair used 0.3
        self.a1 = nn.Linear(32, 2)  # want an action vector output: [log(prob right), log(prob left)]
        self.v1 = nn.Linear(32, 32)
        self.v2 = nn.Linear(32, 1)  # Output the expected return

    def forward(self, obs):
        # in_size = x.size(0)
        # obs = obs.view(-1, 1, 100, 100)  # batch_size x 1 x board_x x board_y
        x = F.relu(self.dp(self.l1(obs)))
        x = F.relu(self.dp(self.l2(x)))
        x = F.relu(self.dp(self.l3(x)))
        x = F.relu(self.dp(self.l4(x)))

        # x = x.view(in_size, -1)  # flatten the tensor
        a = self.a1(self.dp(x))
        action_probs = F.log_softmax(a, dim=-1)  # choose the dimension such that we get something like
        # [exp(-0.6723) +  exp(-0.7144)] = 1 for the output
        v = self.v2(self.dp(self.v1(x)))  # get a linear value for the expected return
        return action_probs, v

    def train_model(self, examples):
        optimizer = optim.Adam(self.parameters()) # , lr=self.args.lr
        action_loss, value_loss, accuracy = [], [], []

        # ------------- CONVERT TO CORRECT DATA TYPE ----------------
        gpu = torch.device("cpu")
        states = torch.tensor(examples[:, 0:4], dtype=torch.float)  # reshapes into a (23002, 4) array
        target_actions = torch.tensor(examples[:, 4], dtype=torch.long)  # reshapes into a (23002, 2) array
        target_returns = torch.tensor(examples[:, 5], dtype=torch.float)

        # if args.cuda:  #if we're using the GPU:
        #    states, target_actions, target_returns = states.contiguous().cuda(), target_actions.contiguous().cuda(), target_returns.contiguous().cuda()
        # states, target_pis, target_vs = Variable(states), Variable(target_actions), Variable(target_returns)
        # We should permute data before batching really. (X is a torch Variable)
        # permutation = torch.randperm(X.size()[0])

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.train()  # set module in training mode
            batch_idx = 0

            for index in range(0, len(target_returns) - self.args.batch_size, self.args.batch_size):

                # -------- GET BATCHES -----------
                # indices = permutation[i:i+batch_size] # [1, 2, 3, 4, 5]  -> [3, 2, 5, 1, 4]
                # target_returns = np.shuffle(target_returns)
                batch_idx = int(index / self.args.batch_size) + 1  # add one so stats print properly
                batch_states = states[index: index + self.args.batch_size]  # torch.Size([64, 4])
                batch_actions = target_actions[index: index + self.args.batch_size]  # torch.Size([64])
                batch_returns = target_returns[index: index + self.args.batch_size]  # torch.Size([64])

                # -------------------- FEED FORWARD ----------------------
                pred_actions, pred_return = self.forward(batch_states)  # torch.Size([64, 2]) and torch.Size([64, 1])
                batch_NumWrong = torch.abs(torch.argmax(pred_actions, dim=1) - batch_actions).sum()

                a_loss = F.nll_loss(pred_actions, batch_actions,
                                    reduction='elementwise_mean') * self.args.pareto  # standard is "elementwise_mean"

                # print(pred_actions.detach(), batch_actions.detach(), a_loss.detach())

                # Suragnair uses tanh for state_values, but their values are E[win] = [-1, 1] where -1 = loss
                # Here we are using the length of time that we have been "up"
                # v_loss = F.binary_cross_entropy(torch.sigmoid(pred_return[:, 0]), torch.sigmoid(batch_returns))
                v_loss = F.mse_loss(pred_return[:, 0], batch_returns, reduction='elementwise_mean')

                action_loss.append(a_loss)
                value_loss.append(v_loss)
                tot_loss = a_loss + v_loss

                # ----------- COMPUTE GRADS AND BACKPROP ----------------
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()

                # --------- PRINT STATS --------------
                # Get array of predicted actions and compare with target actions to compute accuracy

                accuracy.append(1 - (batch_NumWrong.detach().numpy()) / self.args.batch_size)  # counts the different ones
                if batch_idx % 8 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tA-Loss: {:.4f}, V-Loss: {:.4f}\tAccuracy: {:.5f}'.format(
                        epoch + 1,
                        batch_idx * self.args.batch_size,
                        states.size()[0],
                        100 * batch_idx * self.args.batch_size / states.size()[0],
                        a_loss,
                        v_loss,
                        accuracy[batch_idx - 1])

                    )

        return action_loss, value_loss, accuracy  # removed self?

    def test(self, render=False):
        self.eval()
        scores, expected_scores, choices = [], np.zeros(self.args.goal_steps), []

        # ------- PLAY SOME TEST GAMES ----------
        for each_game in range(10):
            self.env.reset()
            score, E_score = 0, []
            game_memory, prev_obs = [], []

            for _ in range(self.args.goal_steps):  # play up to (200) frames
                if render:
                    self.env.render()

                # ----- GENERATE AN ACTION -------
                if len(prev_obs) == 0:  # start by taking a random action
                    action = self.env.action_space.sample()

                else:  # After that take the best predicted action by the neural net
                    x = torch.tensor(prev_obs, dtype=torch.float)
                    action_prob, e_score = self.forward(x)
                    action = np.argmax(action_prob.detach().numpy())
                    E_score.append(
                        e_score.detach().numpy())  # see how the game updates it expected score as we move through

                new_observation, reward, done, info = self.env.step(action)
                prev_obs = new_observation

                # ----- RECORD RESULTS -------
                choices.append(action)  # just so we can work out the ratio of what we're predicting

                game_memory.append([new_observation, action])
                score += reward
                if done: break

            scores.append(score)  # Record the score of each game
            padding = np.zeros(int(self.args.goal_steps - score + 1), dtype=int)
            E_score = np.append([np.array(E_score)], [padding])
            expected_scores = np.vstack((expected_scores, E_score))

        print('Average Score:', sum(scores) / len(scores))
        print('choice 1 (right): {:.4f}  choice 0 (left): {:.4f}'.format(choices.count(1) / len(choices),
                                                                         choices.count(0) / len(choices)))
        print(Counter(scores))

        x = np.linspace(1, len(expected_scores[0]), num=len(expected_scores[0]))
        plt.plot(x, expected_scores[1])
        plt.plot(x, expected_scores[3])
        plt.xlabel("Steps taken")
        plt.ylabel("Expected Return (steps until failure)")
        plt.show()
