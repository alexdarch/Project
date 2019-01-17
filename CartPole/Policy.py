from utils import *
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import csv

# import sys, os, shutil
# import random, math, time
# sys.path.append('..')
#  import argparse
# from torchvision import datasets, transforms
# from pytorch_classification.utils import Bar, AverageMeter

pargs = Utils({
    # ---------- Policy args ------------
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 8,
    'cuda': False, #torch.cuda.is_available(),
    'num_channels': 512,  # 512
    'pareto': 0.2,  # multiply action loss
})


class NetworkArchitecture(nn.Module):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, policy_env):

        self.x_size, self.y_size = policy_env.get_state_2d_size()  # x = pos, y = ang
        self.action_size = policy_env.get_action_size()  # only two calls here
        self.pareto = 0.1

        # torch.cuda.init()   # initialise gpu? necessary?
        # print(torch.cuda.get_device_name(0))

        super(NetworkArchitecture, self).__init__()
        self.conv1 = nn.Conv2d(1, pargs.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(pargs.num_channels, pargs.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(pargs.num_channels, pargs.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(pargs.num_channels, pargs.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(pargs.num_channels)
        self.bn2 = nn.BatchNorm2d(pargs.num_channels)
        self.bn3 = nn.BatchNorm2d(pargs.num_channels)
        self.bn4 = nn.BatchNorm2d(pargs.num_channels)

        self.fc1 = nn.Linear(pargs.num_channels*(self.x_size-4)*(self.x_size-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.x_size, self.x_size)                # batch_size x 1 x board_x x board_y
        # s = s.view(-1, 1 * self.x_size * self.x_size)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, pargs.num_channels*(self.x_size-4)*(self.y_size-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=pargs.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=pargs.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NeuralNet(NetworkArchitecture):
    trains = 0  # count the number of times train_policy is called so we can write csv's

    def __init__(self, policy_env):

        super(NetworkArchitecture, self).__init__()  # won't work without this? but dont know why its neeeded? order of inheritance?
        self.architecture = NetworkArchitecture(policy_env)  # pargs is a global variable so no need to pass in
        self.x_size, self.y_size = policy_env.get_state_2d_size()
        self.action_size = policy_env.get_action_size()

        if pargs.cuda:
            self.architecture.cuda()

    def train_policy(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        optimizer = optim.Adam(self.architecture.parameters())
        a_losses, v_losses, tot_losses = [], [], []

        for epoch in range(pargs.epochs):
            self.architecture.train()    # set module in training mode
            batch_idx, start = 0, time.time()

            while batch_idx < int(len(examples)/pargs.batch_size):
                # --------------- GET BATCHES -------------------
                # randomise the batches (weird that this is done for each batch?
                # why not take first 64 fom pre randomised batch etc)
                sample_ids = np.random.randint(len(examples), size=pargs.batch_size)
                states_2d, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                # convert to torch tensors
                states_2d = torch.FloatTensor(np.array(states_2d).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # -------------- FEED FORWARD -------------------
                if pargs.cuda:
                    print("Using Graphs Card!!!")
                    states_2d, target_pis, target_vs = states_2d.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                states_2d, target_pis, target_vs = Variable(states_2d), Variable(target_pis), Variable(target_vs)

                # -------------- COMPUTE LOSSES -----------------
                out_pi, out_v = self.architecture.forward(states_2d)
                a_loss = self.loss_pi(target_pis, out_pi) * pargs.pareto
                v_loss = self.loss_v(target_vs, out_v)
                total_loss = a_loss + v_loss

                # store losses for writing to file
                a_losses.append(a_loss.detach().numpy().tolist())  # if extend: 'float' object is not iterable
                v_losses.append(v_loss.detach().numpy().tolist())
                tot_losses.append(total_loss.detach().numpy().tolist())

                # ---------- COMPUTE GRADS AND BACK-PROP ------------
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # ------------ TRACK PROGRESS ----------------
                # Get array of predicted actions and compare with target actions to compute accuracy
                batch_idx += 1
                Utils.update_progress(
                    "TRAINING, EPOCH " + str(epoch + 1) + "/" + str(pargs.epochs) + ". PROGRESS OF " + str(
                        int(len(examples) / pargs.batch_size)) + " BATCHES",
                    batch_idx / int(len(examples) / pargs.batch_size), time.time() - start)
                # --------- PRINT STATS --------------
                # if batch_idx % 8 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tA-Loss: {:.4f}, V-Loss: {:.4f}\tAccuracy: {:.5f}'.format(
                #         epoch + 1,
                #         batch_idx * pargs.batch_size,
                #         states_2d.size()[0],
                #         100 * batch_idx * pargs.batch_size / states_2d.size()[0],
                #         a_loss,
                #         v_loss,
                #         accuracy[batch_idx - 1])
                #     )
            if epoch+1 < pargs.epochs:
                print("\r")  # print epoch training on a new line

        # record to CSV file
        print("Action Losses: ", a_losses)
        print("Value Losses: ", v_losses)
        print("tot_losses: ", tot_losses)

        NeuralNet.trains += 1
        losses = [x for x in zip(a_losses, v_losses, tot_losses)]
        with open(r'Data\ActionAndValueLosses'+str(NeuralNet.trains)+'.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(losses)

    def predict(self, state_2d):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                env.get_action_size
            v: a float in [-1,1] that gives the value of the current board
        """
        # preparing input
        state = torch.FloatTensor(state_2d.astype(np.float64))
        if pargs.cuda:
            state = state.contiguous().cuda()

        # state = Variable(state, volatile=True)
        state = state.view(1, self.x_size, self.x_size)

        # print(type(state))
        self.architecture.eval()
        pi, v = self.architecture.forward(state)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
