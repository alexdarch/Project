# import gym
from Controller import Controller
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPoleWrapper

args = Utils({
    # ---------- POLICY ITER ARGS -----------
    'policyIters': 6,
    'numEps': 3,
    'numMCTSSims': 2,
    'testIters': 3,
    'tempThreshold': 15,
    'updateThreshold': 0.95,
    'cpuct': 1,

    'load_model': False,
    'load_folder_file': ('NetCheckpoints', 'best.pth.tar'),

    'policyItersForTrainExamplesHistory': 20,
    'maxlenOfQueue': 200000,
})

if __name__ == "__main__":
    env = CartPoleWrapper()   # equivalent to gym.make("CartPole-v1")
    nnet = nn(env)

    if args.load_model:
        nnet.load_net_architecture(args.load_folder_file[0], args.load_folder_file[1])

    c = Controller(env, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()

    print("Loaded Correctly\n")
    c.policy_iteration()

