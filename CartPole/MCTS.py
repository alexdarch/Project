import math
import numpy as np
EPS = 1e-8


class MCTS():

    def __init__(self, game, nnet, args):  # remove env? Need to re-pass it in every move - not just the first move
        self.env = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, state2D, curr_env, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        observation = curr_env.get_observation()   # obs = [x_pos, x_vel, angle, ang_vel]
        for i in range(self.args.numMCTSSims):
            print("-------- CALL SEARCH -------")
            self.search(state2D, curr_env.reset(observation), done=False)

        s = curr_env.get_state_1d(state2D)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.env.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        curr_env.reset()  # not sure strictly necessary but add in to remove step errors
        return probs

    def search(self, state2D, curr_env, done):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        # possibly replace all self.env with curr_env? to get rid of the calling step thing.
        # also add in more prints...
        s = curr_env.get_state_1d(state2D)

        # ---------------- TERMINAL STATE ---------------
        if done:
            print("        -- TERMINAL STATE --") # 8 spaces
            return 1

        # ------------- EXPLORING FROM A LEAF NODE ----------------------
        # check if the state has been seen before. If not then assign Ps[s]
        # a probability for each action, eg Ps[s1] = [0.25, 0.75] for a = [0(left) 1(right)]
        # Note, we do not take an action here. Just get an initial policy
        # Also get the state value - work this out later
        if s not in self.Ps:
            print("        -- LEAF NODE --")
            self.Ps[s], v = self.nnet.predict(state2D)
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s  # Is this necessary?

            self.Ns[s] = 0
            return v

        cur_best = -float('inf')
        best_act = -1

        # ------------- GET BEST ACTION -----------------------------
        # search through the valid actions and update the UCB for all actions then update best actions
        # pick the action with the highest upper confidence bound
        for a in range(self.env.get_action_size()):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a
        a = best_act

        # ----------- RECURSION TO NEXT STATE ------------------------
        observation, reward, next_done, _ = curr_env.step(a)
        next_s = curr_env.get_state_2d(observation, state2D)
        v = self.search(next_s, curr_env, next_done)

        # ------------ BACKUP Q-VALUES AND N-VISITED -----------------
        # after we reach the terminal condition then the stack unwinds and we
        # propagate up the tree backing up Q and N as we go
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
