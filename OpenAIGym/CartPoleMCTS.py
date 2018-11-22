import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt


class MCTS():

    def __init__(self, env, nnet, args):
        self.statetraj = np.zeros((100, 100))
        self.args = args
        self.env = deepcopy(env)
        self.nnet = nnet    # New policy per policy iteration
                            # (technically reinitialised each episode for PI, but it's the same until it's updated)
        self.c_puct = 0.1
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

    def getActionProb(self, start_state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        the state/step that it was called on (i.e. mcts_env).
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        # ------------ SEARCH THE TREE MANY TIMES -------------
        print("##################### Start Tree Search ####################")
        self.env.reset()
        self.update2Dtraj(start_state[0:4], reset=True)  # reset the state trajectory
        for i in range(self.args.numMCTSSims):
            self.search(start_state[5])            # send the game and if done
            # plt.imshow(self.statetraj)  # , extent=[-2.4, 2.4, -41.8, 41.8]
            # plt.show()

        self.update2Dtraj(start_state[0:4], reset=True)  # reset the state trajectory
        s = self.get1Dtraj()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(0, 2)]  # range(0, 2) = action size
        print(counts)
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]  # /float(sum(counts))
        return probs

    def search(self, done):
        # ---------------- TERMINAL STATE ---------------
        if done:
            print("done")
            return 1
        s = self.get1Dtraj()  # the initial statetraj is defined in getActionProb

        # ------------- EXPLORING FROM A LEAF NODE ----------------------
        # check if the state has a policy from it yet, if not then its a leaf
        # should check if the neural net has assigned a +ve prob to any policy (see suragnair)
        if s not in self.Ps:
            action_prob, expected_v = self.nnet.forward(torch.tensor(self.statetraj, dtype=torch.float))
            #self.Ps[s] = np.exp(action_prob.detach().numpy())
            rint = self.env.action_space.sample()
            if rint == 0:
                self.Ps[s] = np.array([0, 1])
            else:
                self.Ps[s] = np.array([1, 0])

            print("Leaf node Predicted action: ", self.Ps[s])
            self.Ns[s] = 0
            return 1        # Should 1 or v be returned for the reward?

        # ------------- GET BEST ACTION -----------------------------
        # search through the valid actions and update the UCB for all actions then update best actions
        max_u, best_a = -float("inf"), -1
        for a in range(1):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c_puct*self.Ps[s][a]*np.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
            else:
                u = self.c_puct*self.Ps[s][a]*np.sqrt(self.Ns[s] + 1e-8)     # Q = 0 ?

            if u > max_u:
                max_u = u
                best_a = a
        a = best_a
        print("best action is: ", a)

        # ----------- RECURSION TO NEXT STATE ------------------------
        sp, reward, done, info = self.env.step(a)
        self.update2Dtraj(sp)  # updates the state trajectory
        v = self.search(done)  # the recursion adds 1 to v each return
        print("unwinding, v = ", v)

        # ------------ BACKUP Q-VALUES AND N-VISITED -----------------
        # after we reach the terminal condition then the stack unwinds and we
        # propagate up the tree backing up Q and N as we go
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)]*self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)]+1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

    def update2Dtraj(self, curr_state, reset=False):
        # curr_state numpy array [xpos, xvel, angle, angle vel]
        # max values are:         [+-2.4, inf, +-41.8, inf]
        x_edges, y_edges = np.linspace(-2.4, 2.4, 101), np.linspace(-41.8, 41.8, 101)
        new_pos, _, _ = np.histogram2d([curr_state[0], ], [curr_state[2], ], bins=(x_edges, y_edges))
        if reset:
            self.statetraj = new_pos
        else:
            self.statetraj = self.statetraj*0.7 + new_pos

        return self.statetraj

        # plt.imshow(self.statetraj, extent=[-2.4, 2.4, -41.8, 41.8])
        # plt.show()

    def get1Dtraj(self):
        traj1D = np.reshape(self.statetraj, (-1,))    # converts to (10000,) array

        s_ten = torch.tensor(traj1D, dtype=torch.float)

        return tuple(traj1D.tolist())
