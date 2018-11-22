import gym
from Controller import Controller
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPole
import torch

args = dotdict({
    # ---------- POLICY ITER ARGS -----------
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    'num_channels': 512,
    'dropout': 0.3,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    base_env = gym.make("CartPole-v0")
    env = CartPole(base_env)
    nnet = nn(env, args)

    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Controller(env, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()

    print("Loaded Correctly")
    c.policyIteration()

