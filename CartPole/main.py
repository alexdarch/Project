# import gym
from Controller import Controller
from Policy import NeuralNet as nn
from utils import *
from CartPoleWrapper import CartPoleWrapper

args = Utils({
    # ---------- POLICY ITER ARGS -----------
    'policyIters': 20,  # 8
    'initialTrainEps': 40,
    'unopposedTrains': 2,         # how many trains do we ignore the adversary for?
    'trainEps': 40,  # 40,
    'numMCTSSims': 20,  # 15/20,
    'tempThreshold': 7,
    'cpuct': 1.0,
    'mctsTree': False,
    'renderEps': False,


    'checkpoint_folder': "NetCheckpoints",
    'load_model': False,
    'load_folder_file': ('NetCheckpoints', 'best.pth.tar'),

    'numItersForTrainExamplesHistory': 5,
    'maxlenOfQueue': 200000,
})

if __name__ == "__main__":
    env = CartPoleWrapper(adversary=0, unopposedTrains=args.unopposedTrains)   # 0 is nnet adversary, which we always use whilst training
    nnet = nn(env)

    if args.load_model:
        nnet.load_net_architecture(args.load_folder_file[0], args.load_folder_file[1])
        # print("Load trainExamples from file")
        # c.loadTrainExamples()
        print("loaded a nnet")

    c = Controller(env, nnet, args)
    print("Loaded Correctly\n")
    c.policy_iteration()

