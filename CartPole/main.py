# import gym
from Controller import Controller
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPoleWrapper

args = Utils({
    # ---------- POLICY ITER ARGS -----------
    'policyIters': 200,
    'trainEps': 20,  # 20,
    'testEps': 150,   # 15,
    'numMCTSSims': 20,  # 15/20,
    'tempThreshold': 15,
    'updateThreshold': 1.0,  # the best mean needs to be thresh x as good to stay as best
    'cpuct': 1.0,
    'keepAbove': 0,
    'mctsTree': False,

    'checkpoint_folder': "NetCheckpoints",
    'load_model': False,
    'load_folder_file': ('NetCheckpoints', 'best.pth.tar'),

    'numItersForTrainExamplesHistory': 3,
    'maxlenOfQueue': 200000,
})

if __name__ == "__main__":
    env = CartPoleWrapper()   # equivalent to gym.make("CartPole-v1")
    nnet = nn(env)

    if args.load_model:
        nnet.load_net_architecture(args.load_folder_file[0], args.load_folder_file[1])
        print("loaded a nnet")

    c = Controller(env, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()

    print("Loaded Correctly\n")
    c.policy_iteration()

