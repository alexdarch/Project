# import gym
import numpy as np
from utils import *
from MCTS import MCTS
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPoleWrapper
from Compare import Compare

args = Utils({
    # ---------- POLICY ITER ARGS -----------
    'policyIters': 8,  # 8
    'adversary': 0,  # Enumerate: 0:Adversary, 1:None, 2:EveryStepRandom, 3:LargeRandom
    'testEps': 10,   # 10,
    'numMCTSSims': 15,  # 15/20,
    'tempThreshold': 7,
    'updateThreshold': 0,  # the best mean needs to be thresh x as good to stay as best
    'cpuct': 1.0,
    'mctsTree': True,
    'renderTestEps': True,

    'checkpoint_folder': "NetCheckpoints",
    'load_model': False,
    'load_folder_file': ('NetCheckpoints', 'best.pth.tar'),
})

env = CartPoleWrapper(adversary=0)   # equivalent to gym.make("CartPole-v1")
nnet = nn(env)

if args.load_model:
    nnet.load_net_architecture(args.load_folder_file[0], args.load_folder_file[1])
    print("loaded a nnet")

test = Evaluate()
test.evaluate()


class Evaluate:
    def __init__(self):
        self.curr_player_elo = 1000
        self.curr_adv_elo = 1000
        env = CartPoleWrapper(adversary=0)

        # all players
        rp = self.RandomAdversary
        gp = self.NNetAdversary  # ?

    def evaluate(self):

        for checkpoint in ():
            c = Compare(p1, p2)
            c.evaluate_policies(args.testEps)

        # save data at each loop. here, not in the other one

        # nnet players
        n1 = nn(env)
        nnet.load_net_architecture(args.load_folder_file[0], args.load_folder_file[1])

        args1 = Utils({'numMCTSSims': 50, 'cpuct': 1.0})
        challenger_mcts = MCTS(self.env, self.challenger_nnet, self.args)
        current_mcts = MCTS(self.env, self.nnet, self.args)
        player_action = lambda s2d, root, agent: np.argmax(current_mcts.get_action_prob(s2d, root, agent, temp=0))
        adv_action = lambda s2d, root, agent: np.argmax(challenger_mcts.get_action_prob(s2d, root, agent, temp=0))

        eval = Evaluate(self.env, self.args, self.curr_player_elo, self.curr_adv_elo)
        chal_player_elo, chal_adv_elo = eval.evaluate_policies()
        self.curr_player_elo, self.curr_adv_elo = chal_player_elo, chal_adv_elo


    def RandomAdversary(self):
        pass

    def NoAdversary(self):
        pass

    def NNetAdversary(self):
        pass




