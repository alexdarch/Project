import time, csv, os
from utils import *
from Evaluate import Evaluate
from CartPoleWrapper import CartPoleWrapper
from Policy import NeuralNet as nn

args = Utils({
    # ---------- POLICY ITER ARGS -----------

    'adversary': 2,  # Enumerate: 0:TrainingPolicy, 1:TestPolicy, 2:RandomPolicy, 3:None
    'adversaryIter': 0,  # Which adversary to play against if 2 adversary==1
    'testEps': 1,   # 10,
    'numMCTSSims': 15,  # 15/20,
    'cpuct': 1.0,
    'mctsTree': False,  # Doesn't work for results due to rand adv not contributing to tree.
    'renderTestEps': False,

    'policy_folder': os.path.join('..\FinalReport', 'Results', 'Training1D'),  # load the player
    'results_folder': 'TestData',
    'checkpoint_folder': "NetCheckpoints",
})
assert os.path.exists(args.policy_folder), 'folder does not exist'

# just keeping this at the top of the file
env = CartPoleWrapper(adversary=args.adversary, unopposedTrains=0)  # equivalent to gym.make("CartPole-v1")
nnet = nn(env)
test = Evaluate(nnet, env, args)
names = ['Workin', 'AdvF2']  # 'AdvF0', 'AdvF25', 'AdvF50', 'AdvF75', 'AdvF1',
hs = [0.1, 0.2]  # 0, 0.025, 0.05, 0.075, 0.1,
for name, h in zip(names, hs):
    print(name, h)
    test.env.handicap = h
    test.evaluate(name)


