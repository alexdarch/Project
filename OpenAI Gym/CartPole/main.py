import gym
from Controller import Controller
from Policy import Net
from MCTS import MCTS
import numpy as np
''' 
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. 
The pendulum starts upright, and the goal is to prevent it from falling over. 
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the 
cart moves more than 2.4 units from the center.
'''


class dotDict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotDict({
    'lr': 0.0005,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 64, #256,
    'pareto': 5000, # a factor to multiply action loss by to get optimal loss (5000 ish seems to work well)
    'num_channels': 512,
    'goal_steps': 201, #200 is the limit for cart-pole
    'score_requirement': 65,
    'initial_games': 30000,
    'policyUpdates': 2,    #10
    'policyEpisodes': 250, #250
    'numMCTSSims': 50,
})
print(args.pareto)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # observation = env.reset()
    # print(observation)

    nnet = Net(env, args)

    # if args.load_model:
    #   nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # c = Controller(env, args, nnet)
    # best_model = c.policyIteration()
    mcts = MCTS(env, nnet, args)

    probs = mcts.getActionProb(np.array([ 0.01642717,  0.01088803, -0.03848342, -0.00444985, 1, 0]))
    print(probs)


    # best_model.test(render=False)



