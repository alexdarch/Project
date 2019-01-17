# import gym
from Controller import Controller
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPoleWrapper

args = Utils({
    # ---------- POLICY ITER ARGS -----------
    'policyIters': 10,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.95,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 8,
    'testIters': 10,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'policyItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    env = CartPoleWrapper()   # equivalent to gym.make("CartPole-v1")
    nnet = nn(env)

    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Controller(env, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()

    print("Loaded Correctly\n")
    c.policy_iteration()

