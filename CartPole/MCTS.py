import numpy as np
from ete3 import Tree, TreeStyle, TextFace, NodeStyle
from matplotlib import cm  # get colours
EPS = 1e-8
g_accuracy = 1e12


class MCTS:

    def __init__(self, game, nnet, args):  # remove env? Need to re-pass it in every move - not just the first move
        self.env = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        # for plotting a tree diagram
        self.tree = Tree()

    def get_action_prob(self, state_2d, root_state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # observation = curr_env.get_state()   # obs = [x_pos, x_vel, angle, ang_vel]
        # print("Base State: ", curr_env.get_mcts_state(state, g_accuracy))
        for i in range(self.args.numMCTSSims):
            # print("-------- CALL SEARCH -------")
            self.env.reset(root_state)
            self.search(state_2d, done=False)

        self.env.reset(root_state)
        s = self.env.get_mcts_state(root_state, g_accuracy)
        # self.update_tree_values()  # only actually need to update these when showing the tree
        player_counts, adversary_counts = [0]*self.env.get_action_size(), [0]*self.env.get_action_size()
        # print(self.Nsa[(s, 0, 0)], self.Nsa[(s, 1, 0)], self.Nsa[(s, 0, 1)], self.Nsa[(s, 1, 1)])
        for x in range(self.env.get_action_size()):
            # visits = [True if (s, a, x) in self.Qsa else False for x in range(self.env.get_action_size)]
            player_counts = [player_counts[a]+self.Nsa[(s, a, x)] if (s, a, x) in self.Nsa
                             else player_counts[a] for a in range(self.env.get_action_size())]
            adversary_counts = [adversary_counts[a_adv]+self.Nsa[(s, x, a_adv)] if (s, x, a_adv) in self.Nsa
                                else adversary_counts[a_adv] for a_adv in range(self.env.get_action_size())]
        # print(player_counts, adversary_counts)

        if temp == 0:
            bestA = np.argmax(player_counts)
            bestAdv = np.argmax(adversary_counts)
            probsA, probsAdv = [0] * len(player_counts), [0] * len(adversary_counts)
            probsA[bestA] = 1
            probsAdv[bestAdv] = 1
            return probsA, probsAdv

        player_counts = [x ** (1. / temp) for x in player_counts]
        player_probs = [x / float(sum(player_counts)) for x in player_counts]

        adversary_counts = [x ** (1. / temp) for x in adversary_counts]
        adversary_probs = [x / float(sum(adversary_counts)) for x in adversary_counts]
        return player_probs, adversary_probs

    def search(self, state_2d, done):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard

        *** remember to convert v to float if bringing in a new definition
        """
        state = self.env.get_state()
        s = self.env.get_mcts_state(state, g_accuracy)

        # ---------------- TERMINAL STATE ---------------
        if done:
            if self.env.mcts_steps >= self.env.steps_till_done:
                _, _, v = self.nnet.predict(state_2d)
                v = float(v[0])
                # print("done and got to the end at step: ", curr_env.steps, " and value: ", v)
                return v
            # print("Done, at step: ", curr_env.steps, " returning: ", curr_env.terminal_cost)
            # return value as if fallen over
            return float(self.env.terminal_cost * ((self.env.steps_till_done - self.env.mcts_steps)/self.env.steps_till_done + 1))

        # ------------- EXPLORING FROM A LEAF NODE ----------------------
        # check if the state has been seen before. If not then assign Ps[s]
        # a probability for each action, eg Ps[s1] = [0.25, 0.75] for a = [0(left) 1(right)]
        # Note, we do not take an action here. Just get an initial policy
        # Also get the state value - work this out later
        if s not in self.Ps:
            pi_a, pi_adv, v = self.nnet.predict(state_2d)
            self.Ps[s] = (pi_a, pi_adv)   # tuple of np.array: (np.[0.6, 0.4], np.[0.2, 0.8])
            v = float(v[0])
            # since using softmax, pi should already be normed!
            # sum_Ps_s = np.sum(self.Ps[s])
            # self.Ps[s] /= sum_Ps_s  # Normalise probs
            self.Ns[s] = 0
            return v

        # ------------- GET PLAYER AND ADVERSARY ACTIONS -----------------------------
        # search through the valid actions and update the UCB for all actions then update best actions
        # pick the action with the highest upper confidence bound
        action = []
        cur_best = -float('inf')  # set current best ucb to -inf
        best_act = None  # null action
        player = 0
        for a in range(self.env.get_action_size()):
            u_sum = 0
            for a_adv in range(self.env.get_action_size()):
                # for cartpole the actions [0, 1] correspond to [-1, +1], but this is only  resolved in CartPoleWrapper
                if (s, a, a_adv) in self.Qsa:
                    u = self.Qsa[(s, a, a_adv)] + self.args.cpuct * self.Ps[s][player][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a, a_adv)])
                else:
                    u = self.args.cpuct * self.Ps[s][player][a_adv] * np.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                u_sum += u
            u_avg = u_sum/self.env.get_action_size()
            if u_avg > cur_best:
                cur_best = u_avg
                best_act = a
        action.append(best_act)

        cur_best = -float('inf')  # set current best ucb to -inf
        best_act = None  # null action
        player = 1
        for a_adv in range(self.env.get_action_size()):
            u_sum = 0
            for a in range(self.env.get_action_size()):
                # for cartpole the actions [0, 1] correspond to [-1, +1], but this is only  resolved in CartPoleWrapper
                if (s, a, a_adv) in self.Qsa:
                    u = -self.Qsa[(s, a, a_adv)] + self.args.cpuct * self.Ps[s][player][a_adv] * np.sqrt(self.Ns[s]) / (
                                1 + self.Nsa[(s, a, a_adv)])
                else:
                    u = self.args.cpuct * self.Ps[s][player][a] * np.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                u_sum += u
            u_avg = u_sum / self.env.get_action_size()
            if u_avg > cur_best:  # choose the lowest u for the adversary
                cur_best = u_avg
                best_act = a_adv

        action.append(best_act)  # action = [best_player_act, best_adv_act], where act is an elm of {0, 1} here
        #print("actions: ", action)

        # ----------- RECURSION TO NEXT STATE ------------------------
        next_state, loss, next_done, _ = self.env.step(*action)      # not a true step -> will update mcts_steps
        if self.args.mctsTree:
            self.add_tree_node(state, next_state, action)
        next_state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
        v = self.search(next_state_2d, next_done)

        # ------------ BACKUP Q-VALUES AND N-VISITED -----------------
        # after we reach the terminal condition then the stack unwinds and we
        # propagate up the tree backing up Q and N as we go
        a, adv = action
        if (s, a, adv) in self.Qsa:
            self.Qsa[(s, a, adv)] = (self.Nsa[(s, a, adv)] * self.Qsa[(s, a, adv)] + v) / (self.Nsa[(s, a, adv)] + 1)
            self.Nsa[(s, a, adv)] += 1

        else:
            self.Qsa[(s, a, adv)] = v
            self.Nsa[(s, a, adv)] = 1

        self.Ns[s] += 1
        return v

    def add_tree_node(self, prev_state, curr_state, actions):
        """
        Updates the tree with each new node's
            "name" - the mcts_state e.g. (699923423, 124134235, 234234124, 23524634634)
            "state" - the observation rounded to 2dp
            "action" - the action needed to get from prev_state to the curr_state
        Note, normally it is done with (s, a) = (curr_state, action_taken_from_s)... However, this would mean the tree
        has to be 2x as big. Instead use (curr_state, action_to_get_to_s). Luckily, ete3 always puts +1 on the left!
        """
        child_state = self.env.round_state(curr_state)
        parent_mcts_state = self.env.get_mcts_state(prev_state, g_accuracy)
        child_mcts_state = self.env.get_mcts_state(curr_state, g_accuracy)

        prev_len = len(self.tree.search_nodes(name=parent_mcts_state))
        curr_len = len(self.tree.search_nodes(name=child_mcts_state))

        # if neither the parent or the child node exist then this must be a root
        if curr_len == 0 and prev_len == 0:
            parent = self.tree.add_child(name=parent_mcts_state, dist=5)
            child = parent.add_child(name=child_mcts_state, dist=5)
            parent.add_features(state=self.env.round_state(prev_state), action=0)  # the root node doesn't have an action that got to it
            child.add_features(state=child_state, action=actions)
            return

        # if there are no nodes with curr_mcts_state already in existence then find the parent and add a child
        parent = self.tree.search_nodes(name=parent_mcts_state)[0]
        if len(self.tree.search_nodes(name=child_mcts_state)) == 0:
            child = parent.add_child(name=child_mcts_state, dist=5)
            child.add_features(state=child_state, action=actions)

        # if child and parent already exist then this must be another simulation -> dont need to add another node
        # if we have just taken the next true step, then change the node style
        if self.env.mcts_steps <= self.env.steps+1:
            nstyle = NodeStyle()
            nstyle["size"] = 0
            nstyle["vt_line_color"] = "#ff0000"
            nstyle["vt_line_width"] = 8
            nstyle["vt_line_type"] = 0  # 0 solid, 1 dashed, 2 dotted
            nstyle["shape"] = "sphere"
            nstyle["size"] = 20
            nstyle["fgcolor"] = "darkred"
            parent.add_features(steps=self.env.steps)
            parent.set_style(nstyle)

    def update_tree_values(self):
        # after each mcts need to update each node with Qsa, Nsa etc with the new values
        # then decide what ones to colour
        tree_itr = self.tree.traverse()
        next(tree_itr)   # skip the first two nodes the root and the root's root (not sure why there's 2?)
        next(tree_itr)
        for node in tree_itr:
            s, a = node.up.name, node.action
            a, a_adv = a
            state = [float(dim) / g_accuracy for dim in node.name]

            # note that (s, a) is the same as node.name (since parent state & action taken from there = child state)
            try:
                delattr(node, "_faces")     # need to remove previously added faces otherwise they stack
            except:
                pass

            # -------- ADD ANNOTATION FOR STATE LOSS AND ACTION TAKEN ----------
            loss = self.env.state_loss(state=state)
            loss_face = TextFace("(s, a, a_adv) = ({0}, {1}),  v = {2:.3f}".format(self.env.round_state(state=state), (a, a_adv), loss))
            c_loss = cm.viridis(255+int(loss*255))  # viridis goes from 0-255
            c_loss = "#{0:02x}{1:02x}{2:02x}".format(*[int(round(i * 255)) for i in [c_loss[0], c_loss[1], c_loss[2]]])
            loss_face.background.color = c_loss   # need rgb colour in hex, "#FFFFFF"=(255, 255, 255)
            node.add_face(loss_face, column=0, position="branch-top")

            # -------- ADD ANNOTATION FOR ACTION VALUE WRT PLAYER, Q --------
            ucb_p = self.args.cpuct * self.Ps[s][0][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a, a_adv)])
            # print(self.Qsa[(s, a)], ucb, ucb+self.Qsa[(s, a)])
            QP_face = TextFace("u_player = {0:.3f}(Q(s))+{1:.3f}(ucb)= {2:.3f}".format(self.Qsa[(s, a, a_adv)], ucb_p, ucb_p+self.Qsa[(s, a, a_adv)]))
            c_value = cm.viridis(255+int((self.Qsa[(s, a, a_adv)]+ucb_p)*255))  # plasma goes from 0-255
            c_value = "#{0:02x}{1:02x}{2:02x}".format(*[int(round(i * 255)) for i in [c_value[0], c_value[1], c_value[2]]])
            QP_face.background.color = c_value
            node.add_face(QP_face, column=0, position="branch-bottom")

            # -------- ADD ANNOTATION FOR ACTION VALUE WRT PLAYER, Q --------
            ucb_a = self.args.cpuct * self.Ps[s][1][a_adv] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a, a_adv)])
            # print(self.Qsa[(s, a)], ucb, ucb+self.Qsa[(s, a)])
            QA_face = TextFace("u_adv = {0:.3f}(Q(s))+{1:.3f}(ucb)= {2:.3f}".format(self.Qsa[(s, a, a_adv)], ucb_a, ucb_a+self.Qsa[(s, a, a_adv)]))
            c_value = cm.viridis(255+int((self.Qsa[(s, a, a_adv)]+ucb_a)*255))  # plasma goes from 0-255
            c_value = "#{0:02x}{1:02x}{2:02x}".format(*[int(round(i * 255)) for i in [c_value[0], c_value[1], c_value[2]]])
            QA_face.background.color = c_value
            node.add_face(QA_face, column=0, position="branch-bottom")

            # -------- ADD ANNOTATION FOR NUMBER OF VISITS -------
            try:
                N_face = TextFace(" Nsa(s_par, a, a_adv)={}, Ns(s)={}".format(self.Nsa[(s, a, a_adv)], self.Ns[node.name]))
            except:
                N_face = TextFace(" Nsa(s_par, a, a_adv)={}".format(self.Nsa[(s, a, a_adv)]))
            c_vis = cm.YlGn(int(self.Nsa[(s, a, a_adv)]))  # YlGn goes from 0-255
            c_vis = "#{0:02x}{1:02x}{2:02x}".format(*[int(round(i * 255)) for i in [c_vis[0], c_vis[1], c_vis[2]]])
            N_face.background.color = c_vis
            node.add_face(N_face, column=1, position="branch-bottom")

    def show_tree(self, values=True):
        self.update_tree_values()  # only actually need to update these when showing the tree

        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.show_branch_support = False
        # ts.rotation = 90
        # ts.title.add_face(TextFace("Hello ETE", fsize=20), column=0)
        # each node contains 3 attributes: node.dist, node.name, node.support
        self.tree.show(tree_style=ts)  # , show_internal=True)
