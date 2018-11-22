

from utils import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import sys, os, shutil
# import random, math, time
# sys.path.append('..')
#  import argparse
# from torchvision import datasets, transforms
# from pytorch_classification.utils import Bar, AverageMeter

pargs = dotdict({
    # ---------- Policy args ------------
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,  # torch.cuda.is_available(),
    'num_channels': 512,
})


class NeuralNet(nn.Module):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, policy_env, args):

        self.x_size, self.y_size = policy_env.get2DstateSize()
        self.action_size = policy_env.getActionSize()
        self.args = args
        self.pareto = 1

        # torch.cuda.init()   # initialise gpu? necessary?
        print(torch.cuda.get_device_name(0))

        super(NeuralNet, self).__init__()
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
        optimizer = optim.Adam(self.parameters())

        for epoch in range(pargs.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.train()    # set module in training mode
            batch_idx = 0
            accuracy = []

            while batch_idx < int(len(examples)/pargs.batch_size):
                # -------- GET BATCHES -----------
                # randomise the batches
                sample_ids = np.random.randint(len(examples), size=pargs.batch_size)
                states2D, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                #convert to torch tensors
                states2D = torch.FloatTensor(np.array(states2D).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # -------------- FEED FORWARD -------------------
                if pargs.cuda:
                    states2D, target_pis, target_vs = states2D.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                states2D, target_pis, target_vs = Variable(states2D), Variable(target_pis), Variable(target_vs)

                # ------------- COMPUTE LOSSES --------------
                out_pi, out_v = self.forward(states2D)
                a_loss = self.loss_pi(target_pis, out_pi) * self.pareto
                v_loss = self.loss_v(target_vs, out_v)
                total_loss = a_loss + v_loss

                batch_NumWrong = torch.abs(torch.argmax(out_pi, dim=1) - out_pi).sum()

                # ---------- COMPUTE GRADS AND BACKPROP ------------
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # --------- PRINT STATS --------------
                # Get array of predicted actions and compare with target actions to compute accuracy

                accuracy.append(1 - (batch_NumWrong.detach().numpy()) / pargs.batch_size)  # counts the different ones
                if batch_idx % 8 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tA-Loss: {:.4f}, V-Loss: {:.4f}\tAccuracy: {:.5f}'.format(
                        epoch + 1,
                        batch_idx * pargs.batch_size,
                        states2D.size()[0],
                        100 * batch_idx * pargs.batch_size / states2D.size()[0],
                        a_loss,
                        v_loss,
                        accuracy[batch_idx - 1])
                    )

    @torch.no_grad()  # should wrap function in this?
    def predict(self, state_2D):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # preparing input
        state = torch.FloatTensor(state_2D.astype(np.float64))
        if pargs.cuda:
            state = state.contiguous().cuda()

        # or instead of the decorator use this ?:
        # with torch.no_grad():
        #   state = state.view...
        #   self.eval()
        #   pi, v = self.forward(state)

        # state = Variable(state, volatile=True)
        state = state.view(1, self.x_size, self.x_size)

        # print(type(state))
        self.eval()
        pi, v = self.forward(state)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
