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
        self.Usa = {}

        # for plotting a tree diagram
        self.tree = Tree()

    def get_action_prob(self, state_2d, root_state, agent, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # no point in doing mcts if its random anyways
        for i in range(self.args.numMCTSSims):
            self.env.reset(root_state)
            self.search(state_2d, agent, done=False)

        self.env.reset(root_state)
        s = self.env.get_mcts_state(root_state, g_accuracy)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.env.get_action_size(agent))]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, state_2d, agent, done):
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
        state for the current agent, then its value is -v for the other agent.
        Returns:
            v: the negative of the value of the current canonicalBoard

        *** remember to convert v to float if bringing in a new definition
        """
        agent %= 2
        state = self.env.get_state()
        s = self.env.get_mcts_state(state, g_accuracy)

        # ---------------- TERMINAL STATE ---------------
        if done:
            if self.env.mcts_steps >= self.env.steps_till_done:
                _, v = self.nnet.predict(state_2d, agent)
                # print("done and got to the end at step: ", curr_env.steps, " and value: ", v)
                return v
            # print("Done, at step: ", curr_env.steps, " returning: ", curr_env.terminal_cost)
            # return value as if fallen over
            t = float(self.env.terminal_cost) # * ((self.env.steps_till_done - self.env.mcts_steps)/self.env.steps_till_done + 1))
            return t

        # ------------- EXPLORING FROM A LEAF NODE ----------------------
        # check if the state has been seen before. If not then assign Ps[s]
        # a probability for each action, eg Ps[s1] = [0.25, 0.75] for a = [0(left) 1(right)]
        # Note, we do not take an action here. Just get an initial policy
        # Also get the state value - work this out later
        if s not in self.Ps:
            pi, v = self.nnet.predict(state_2d, agent)
            self.Ps[s] = pi   # list
            self.Ns[s] = 0
            return v

        # ------------- GET agent AND ADVERSARY ACTIONS -----------------------------
        # search through the valid actions and update the UCB for all actions then update best actions
        # pick the action with the highest upper confidence bound
        cur_best = -float('inf')  # set current best ucb to -inf
        best_act = None  # null action
        for a in range(self.env.get_action_size(agent)):
            if (s, a) in self.Qsa:
                q = self.Qsa[(s, a)] if agent == 0 else -self.Qsa[(s, a)]
                u = q + self.args.cpuct * self.Ps[s][a]*np.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a]*np.sqrt(self.Ns[s] + EPS)
            self.Usa[(s, a)] = u
            if u > cur_best:
                cur_best = u
                best_act = a
        a = best_act
        # ----------- RECURSION TO NEXT STATE ------------------------
        next_state, loss, next_done, _ = self.env.step(a, agent)      # not a true step -> will update mcts_steps
        if self.args.mctsTree:
            self.add_tree_node(state, next_state, a, agent)
        next_state_2d = self.env.get_state_2d(prev_state_2d=state_2d)
        v = self.search(next_state_2d, agent+1, next_done)

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

    def add_tree_node(self, prev_state, curr_state, a, agent):
        """
        Updates the tree with each new node's
            "name, s_t" - the mcts_state e.g. (699923423, 124134235, 234234124, 23524634634)
            "agent, p_t" - the agent at time t
            "action, a_(t-1)" - the action needed to get from prev_state to the curr_state (the previous agents action)
        Note, normally a node is stored as (s, a) = (curr_state, action from curr_state)...
        However, this would mean the tree has to be 2x as big. Instead use (curr_state, action to curr_state).
        Luckily, ete3 always puts +1 on the left!
        """
        parent_mcts_state = self.env.get_mcts_state(prev_state, g_accuracy)
        child_mcts_state = self.env.get_mcts_state(curr_state, g_accuracy)

        prev_len = len(self.tree.search_nodes(name=parent_mcts_state))
        curr_len = len(self.tree.search_nodes(name=child_mcts_state))

        # note since the action is taken from prev->curr, store agent as the agent that took the action to get to the current state.
        # if neither the parent or the child node exist then this must be a root
        if curr_len == 0 and prev_len == 0:
            parent = self.tree.get_tree_root()
            parent.name = parent_mcts_state; parent.dist = 5
            child = parent.add_child(name=child_mcts_state, dist=5)
            parent.add_features(action=None, agent=agent)  # the root node doesn't have an action that got to it
            child.add_features(action=a, agent=(agent+1) % 2)
            return

        # if there are no nodes with curr_mcts_state already in existence then find the parent and add a child
        parent = self.tree.search_nodes(name=parent_mcts_state)[0]
        if len(self.tree.search_nodes(name=child_mcts_state)) == 0:
            child = parent.add_child(name=child_mcts_state, dist=5)
            child.add_features(action=a, agent=(agent+1) % 2)

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
        """
        Update the tree created in add_tree_node with values of Nsa, Ns, Ps, Qsa and Usa. Colour each of the nodes
        with a colour representing their values.
        Note that nodes on the tree are represented as (s_t, a_(t-1), p_t) which means that when calling Qsa, Usa
        or Nsa (ones involving a) we need to use (s_(t-1), a_(t-1), p_(t-1)), as opposed to Ns and Ps, where we use
        (s_t, p_t).
        """
        tree_itr = self.tree.traverse()
        root = next(tree_itr)   # skip the first two nodes the root and the root's root (not sure why there's 2?)
        root_face = TextFace(u'(s\u209C, a\u209C\u208B\u2081, agent\u209C) = ({}, {}, {}, Ns={})'.format(
            [float(dim) / g_accuracy for dim in root.name],
            root.action, root.agent, self.Ns[root.name]))
        root_face.background.color = '#FF0000'
        root.add_face(root_face, column=0, position="branch-top")

        for node in tree_itr:
            s, a, agent = node.up.name, node.action, node.up.agent
            p_from_a = node.agent
            state = [float(dim) / g_accuracy for dim in node.name]
            # note that (s, a) is the same as node.name (since parent state & action taken from there = child state)
            try:
                delattr(node, "_faces")     # need to remove previously added faces otherwise they stack
            except:
                pass
            _, v = self.nnet.predict(np.array(state), p_from_a)

            # -------- ADD ANNOTATION FOR STATE LOSS AND ACTION TAKEN ----------
            loss = self.env.state_loss(state=state)
            # unicode to get subscripts
            loss_face = TextFace(u'(x\u209C, u\u209C\u208B\u2081, agent\u209C) = ({0}, {1}, {2}),  c(x\u209C) = {3:.3f}, v_pred = {4:.3f}'.format(self.env.round_state(state=state), a, p_from_a, loss, v))
            c_loss = cm.viridis(255+int(loss*255))  # viridis goes from 0-255
            c_loss = "#{0:02x}{1:02x}{2:02x}".format(*[int(round(i * 255)) for i in [c_loss[0], c_loss[1], c_loss[2]]])
            loss_face.background.color = c_loss   # need rgb colour in hex, "#FFFFFF"=(255, 255, 255)
            node.add_face(loss_face, column=0, position="branch-top")

            # -------- ADD ANNOTATION FOR ACTION VALUE WRT agent, Q --------
            #print("s={}, a={}, agent={}".format(s, a, agent))
            if (s, a) in self.Qsa:

                #print(self.Ns[s])
                #print(self.Nsa[(s, a)])
                #print(self.Ps[s][a])
                q = self.Qsa[(s, a)] if agent == 0 else -self.Qsa[(s, a)]
                ucb = self.args.cpuct * self.Ps[s][a]*np.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
                u = self.Usa[(s, a)]
            else:
                q = 0
                ucb = self.args.cpuct * self.Ps[node.name][a]*np.sqrt(self.Ns[s] + EPS)
                u = ucb
            q_formula = '(Q(x))' if agent == 0 else '(-Q(x))'
            QA_face = TextFace("U\u209C = {:.3f}{} + {:.3f}(ucb) = {:.3f}".format(q, q_formula, ucb, u))

            c_value = cm.viridis(255+int((q+ucb)*255))  # plasma goes from 0-255
            c_value = "#{0:02x}{1:02x}{2:02x}".format(
                *[int(round(i * 255)) for i in [c_value[0], c_value[1], c_value[2]]])
            QA_face.background.color = c_value
            node.add_face(QA_face, column=0, position="branch-bottom")

            # -------- ADD ANNOTATION FOR NUMBER OF VISITS -------
            # have to use node.name for printing Ns
            ns = 0 if node.name not in self.Ns else self.Ns[node.name]
            N_face = TextFace(" Nsa(x\u209C\u208B\u2081, u\u209C)={}, Ns(x)={}".format(self.Nsa[(s, a)], ns))

            c_vis = cm.YlGn(int(255*(1-self.Nsa[(s, a)]/(self.args.numMCTSSims*2))))  # YlGn goes from 0-255
            c_vis = "#{0:02x}{1:02x}{2:02x}".format(*[int(round(i * 255)) for i in [c_vis[0], c_vis[1], c_vis[2]]])
            N_face.background.color = c_vis
            node.add_face(N_face, column=1, position="branch-bottom")

    def show_tree(self):
        self.update_tree_values()  # only actually need to update these when showing the tree

        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.show_branch_support = False
        # ts.rotation = 90
        # ts.title.add_face(TextFace("Hello ETE", fsize=20), column=0)
        # each node contains 3 attributes: node.dist, node.name, node.support
        self.tree.show(tree_style=ts)  # , show_internal=True)
