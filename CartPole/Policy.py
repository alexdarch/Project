from utils import *
import time, csv, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from memory_profiler import profile

pargs = Utils({
    # ---------- Policy args ------------
    'lr': 0.0005,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,  # 512
    'pareto': 0.1  # multiply action loss
})

class ConvNetworkArchitecture(nn.Module):
    def __init__(self, policy_env):
        self.x_size, self.y_size = policy_env.get_state_2d_size()  # x = pos, y = ang
        self.action_size = policy_env.get_action_size(0)
        print("Conv Network Architecture")

        super(ConvNetworkArchitecture, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(10 * 10 * 256, 1600)
        self.fc_bn1 = nn.BatchNorm1d(1600)
        self.action_layer = nn.Linear(1600, self.action_size)
        self.value_layer = nn.Linear(1600, 1)

    def forward(self, x):
        s = x.view(-1, 1, self.x_size, self.y_size)  # converts [1, 40, 40] -> [1, 1, 40, 40]
        s = self.layer1(s)     # [1, 1, 40, 40] -> [1, 64, 20, 20]
        s = self.layer2(s)     # [1, 64, 20, 20] -> [1, 256, 10, 10]
        s = s.reshape(s.size(0), -1)  # [1, 256, 10, 10] -> [1, 25600]
        s = self.drop_out(s)   # [1, 25600] -> [1, 25600]
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=pargs.dropout, training=self.training)  # [1, 25600] -> [1, 1600]

        pi = self.action_layer(s)  # batch_size x action_size
        v = self.value_layer(s)

        return F.softmax(pi, dim=1), -1.0*torch.sigmoid(v)

class StatePlayerNetworkArchitecture(torch.nn.Module):
    def __init__(self, policy_env):
        super().__init__()
        print("State Network Architecture")
        self.action_size = policy_env.get_action_size(0)  # o
        HIDDEN = 400
        self.linear1 = torch.nn.Linear(4, HIDDEN)
        self.linear2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.linear3 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.value_layer = nn.Linear(HIDDEN, 1)
        self.action_layer = nn.Linear(HIDDEN, self.action_size)

    def forward(self, x):
        # takes a vector [action, obs0, obs1, obs2, obs3] and spits out a predicted state
        s = x.view(-1, 4)  # converts [1, 25, 25] -> [1, 1, 25, 25]
        s = self.linear1(s)     # [1, 1, 25, 25] -> [1, 32, 12, 12]
        s = self.linear2(s)     # [1, 32, 12, 12] -> [1, 64, 6, 6]
        s = self.linear3(s)        # [1, 2304] -> [1, 1000]
        pi = self.action_layer(s)
        v = self.value_layer(s)  # batch_size x 1
        return F.softmax(pi, dim=1), -1.0*torch.sigmoid(v)

class StateAdversaryNetworkArchitecture(torch.nn.Module):

    def __init__(self, policy_env):
        super().__init__()
        print("State Network Architecture")
        self.action_size = policy_env.get_action_size(1)
        HIDDEN = 400
        self.linear1 = torch.nn.Linear(4, HIDDEN)
        self.linear2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.linear3 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.value_layer = nn.Linear(HIDDEN, 1)
        self.adversary_layer = nn.Linear(HIDDEN, self.action_size)

    def forward(self, x):
        # takes a vector [action, obs0, obs1, obs2, obs3] and spits out a predicted state
        s = x.view(-1, 4)  # converts [1, 25, 25] -> [1, 1, 25, 25]
        s = self.linear1(s)     # [1, 1, 25, 25] -> [1, 32, 12, 12]
        s = self.linear2(s)     # [1, 32, 12, 12] -> [1, 64, 6, 6]
        s = self.linear3(s)        # [1, 2304] -> [1, 1000]
        pi = self.adversary_layer(s)
        v = self.value_layer(s)  # batch_size x 1
        return F.softmax(pi, dim=1), -1.0*torch.sigmoid(v)


class StateAgentNetworkArchitecture(torch.nn.Module):
    def __init__(self, policy_env):
        super().__init__()
        print("Agent Network Architecture")

        self.action_size = policy_env.get_action_size(0)  # o
        self.linear1 = torch.nn.Linear(4, 200)
        self.fc_bn1 = nn.BatchNorm1d(200)

        self.linear2 = torch.nn.Linear(200, 400)
        self.fc_bn2 = nn.BatchNorm1d(400)

        self.linear3 = torch.nn.Linear(400, 200)
        self.fc_bn3 = nn.BatchNorm1d(200)

        self.value_layer = nn.Linear(200, 1)
        self.action_layer = nn.Linear(200, self.action_size)

    def forward(self, x):
        # takes a vector [action, obs0, obs1, obs2, obs3] and spits out a predicted state
        s = x.view(-1, 4)  # converts [1, 25, 25] -> [1, 1, 25, 25]
        s = F.dropout(F.relu(self.fc_bn1(self.linear1(s))), p=pargs.dropout, training=self.training)  # batch_size x 512
        s = F.dropout(F.relu(self.fc_bn2(self.linear2(s))), p=pargs.dropout, training=self.training)  # batch_size x 512
        s = F.dropout(F.relu(self.fc_bn3(self.linear3(s))), p=pargs.dropout, training=self.training)  # batch_size x 512
        pi = self.action_layer(s)
        v = self.value_layer(s)  # batch_size x 1
        return F.softmax(pi, dim=1), -1.0*torch.sigmoid(v)


class NeuralNet:
    trains = 0  # count the number of times train_policy is called so we can write csv's

    def __init__(self, policy_env):

        self.player_architecture = StateAgentNetworkArchitecture(policy_env)  # ConvNetworkArchitecture(policy_env)  #
        self.adversary_architecture = StateAgentNetworkArchitecture(policy_env)  # ConvNetworkArchitecture(policy_env) #
        self.x_size, self.y_size = policy_env.get_state_2d_size()
        self.action_size = policy_env.get_action_size(0)
        self.unopposedTrains = policy_env.unopposedTrains

        if pargs.cuda:
            self.player_architecture.cuda()
            self.adversary_architecture.cuda()

    def train_policy(self, examples):
        examples = np.array(examples)
        if NeuralNet.trains < self.unopposedTrains:
            architectures = [self.player_architecture]
            optimizers = [optim.Adam(self.player_architecture.parameters(), lr=pargs.lr)]
        else:
            architectures = [self.player_architecture, self.adversary_architecture]
            optimizers = [optim.Adam(self.player_architecture.parameters(), lr=pargs.lr),
                          optim.Adam(self.adversary_architecture.parameters(), lr=pargs.lr)]
        agent_losses = [[], [], [], [], [], []]

        for agent in range(len(architectures)):
            agent_examples = examples[examples[:, 2] == agent, :]
            architectures[agent].train()    # set module in training mode
            batch_idx, start = 0, time.time()
            while batch_idx < int(len(agent_examples)/pargs.batch_size):

                # --------------- GET BATCHES -------------------
                # Stochastic gradient descent -> pick a random sample
                sample_ids = np.random.randint(len(agent_examples), size=pargs.batch_size)
                states_2d, pis, agents, vs = list(zip(*[agent_examples[i] for i in sample_ids]))

                # convert to torch tensors. Players is irrelevent since each nnet is its own player
                states_2d = torch.FloatTensor(np.array(states_2d).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs))

                # -------------- FEED FORWARD -------------------
                if pargs.cuda:
                    states_2d = states_2d.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pis, out_vs = architectures[agent](states_2d)
                # -------------- COMPUTE LOSSES -----------------
                pi_loss = self.loss_pi(target_pis, out_pis)*pargs.pareto
                v_loss = self.loss_v(target_vs, out_vs)
                tot_loss = v_loss + pi_loss

                # ---------- COMPUTE GRADS AND BACK-PROP ------------
                tot_loss.backward()
                optimizers[agent].step()
                optimizers[agent].zero_grad()

                # store losses for writing to file
                if pargs.cuda:
                    pi_loss = pi_loss.cpu()
                    v_loss = v_loss.cpu()
                    # tot_loss = tot_loss.cpu()
                agent_losses[agent*2 + 0].append(pi_loss.detach().numpy().tolist())
                agent_losses[agent*2 + 1].append(v_loss.detach().numpy().tolist())
                # agent_losses[agent*3 + 2].append(tot_loss.detach().numpy().tolist())

                # ------------ TRACK PROGRESS ----------------
                # Get array of predicted actions and compare with target actions to compute accuracy
                batch_idx += 1
                tag = "TRAINING, EPOCH " + str(1) + "/" + str(pargs.epochs) + ". PROGRESS OF " + str(
                        int(len(agent_examples) / pargs.batch_size)) + " BATCHES"
                Utils.update_progress(tag, batch_idx / int(len(agent_examples) / pargs.batch_size), time.time() - start)

        # record to CSV file
        # losses = list(zip(a_losses, a_adv_losses, v_losses, tot_losses))
        with open(r'Data\TrainingData\NNetLosses'+str(NeuralNet.trains)+'.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(agent_losses)
        NeuralNet.trains += 1

    def predict(self, state_2d, agent):
        """
        Input:
            board: current board in its canonical form.
        Returns:
            pi: a policy vector for the current board- a numpy array of length
                env.get_action_size
            v: a float in [-1,1] that gives the value of the current board
        """
        assert agent == 0 or agent == 1, 'Not a valid agent'
        architectures = [self.player_architecture, self.adversary_architecture]
        # preparing input
        state = torch.FloatTensor(state_2d.astype(np.float64))
        if pargs.cuda:
            state = state.contiguous().cuda()
        state = state.view(1, self.x_size, self.y_size)

        architectures[agent].eval()
        pi, v = architectures[agent].forward(state)
        pi, v = pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        pi, v = pi.tolist(), v.tolist()[0]
        #pi = [+100, +100] if agent == 1 else pi
        #v = -100 if agent == 1 else v

        return pi, v

    def loss_pi(self, targets, outputs):
        # the outputs are ln(p) already from log_softmax
        return -torch.sum(targets*outputs)/targets.size()[0]
        # return torch.sum((targets.view(-1) - outputs.view(-1)) ** 2) / (targets.size()[0]*self.action_size)

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_net_architecture(self, folder='NetCheckpoints', filename='checkpoint.pth.tar'):
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)

        player_path = os.path.join(folder, "player"+filename)
        torch.save({
            'state_dict': self.player_architecture.state_dict(),
        }, player_path)
        adversary_path = os.path.join(folder, "adversary"+filename)
        torch.save({
            'state_dict': self.adversary_architecture.state_dict(),
        }, adversary_path)

    def load_net_architecture(self, folder='NetCheckpoints', filename='checkpoint.pth.tar'):
        map_location = None if pargs.cuda else 'cpu'

        player_path = os.path.join(folder, "player"+filename)
        player_checkpoint = torch.load(player_path, map_location=map_location)
        self.player_architecture.load_state_dict(player_checkpoint['state_dict'])

        adversary_path = os.path.join(folder, "adversary"+filename)
        adversary_checkpoint = torch.load(adversary_path, map_location=map_location)
        self.adversary_architecture.load_state_dict(adversary_checkpoint['state_dict'])

