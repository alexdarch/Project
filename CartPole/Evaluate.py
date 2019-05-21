# import gym
import numpy as np
from utils import *
from MCTS import MCTS
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPoleWrapper
from Compare import Compare
import os

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


def play():
    # just keeping this at the top of the file
    env = CartPoleWrapper(adversary=0)  # equivalent to gym.make("CartPole-v1")
    nnet = nn(env)
    test = Evaluate(nnet, env)
    test.evaluate()


class Evaluate:
    def __init__(self, nnet, env):
        self.curr_player_elo = 1000
        self.curr_adv_elo = 1000
        self.env = env
        self.nnet = nnet

        # save data at each loop. here, not in the other one
        args1 = Utils({'numMCTSSims': 50, 'cpuct': 1.0})
        # all players
        self.RandomAdversary = lambda x:
        self.NNetAdversary  # ?

    def evaluate(self):

        iter = 0
        while os.path.exists(r'NetCheckpoints/checkpoint_'+str(iter)+'.pth.tar'):
            self.nnet.load_net_architecture('NetCheckpoints', 'checkpoint_'+str(iter)+'.pth.tar')  # load both agents nnets

            mcts = MCTS(self.env, self.nnet, self.args)
            action = lambda s2d, root, agent: np.argmax(mcts.get_action_prob(s2d, root, agent, temp=0))
            
            c = Compare(p1, p2, self.env, self.args, self.curr_player_elo, self.curr_adv_elo)
            c.evaluate_policies(args.testEps)
            self.curr_player_elo, self.curr_adv_elo = chal_player_elo, chal_adv_elo

    def RandomAdversary(self):
        pass

    def NoAdversary(self):
        pass

    def NNetAdversary(self):
        pass


play()

