import numpy as np
import game
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

class PacmanSimpleDQN(game.Agents):
    def __init__(self, args):
        print("Initialize Pacman Agent (Simple DQN)")
        self.memory_size = args.memory_size # default : 10
        self.state_size = args.state_size # default : 7 (7x7 matrix)
        self.Qfunction = SimpleDQN()
        self.replay_mem = deque() 
        self.train_size = args.train_size # default : 50

    def registerInitialState
    # run once at the beginning. Initialization of the Model should not be in here
    # no return


    def observationFunction
    # run while not game over
    # return observation

    def getAction(self, state):
        values = self.Qfunction(util.getState(state, self.state_size))

        # softmax function for choosing the next action
        dist = util.Counter()
        for a in state.getLegalActions(0):
            if a == 'NORTH':
                dist[a] = np.exp(values[0])
            elif a == 'SOUTH':
                dist[a] = np.exp(values[1])
            elif a == 'EAST':
                dist[a] = np.exp(values[2])
            elif a == 'WEST':
                dist[a] = np.exp(values[3])
            elif a == 'STOP':
                dist[a] = np.exp(values[4])
            else:
                raise(BaseException('PacmanSimpleDQN : wrong legal Action'))
        dist.normalize()

        return util.chooseFromDistribution(dist)

    def final:
    # run after the game over
    # TODO : print stats // do i need this??

    def train(self):
        # generate train batch (from replay memory)
        dataset = np.random.choice(self.replay_mem, self.train_size, False)

        # train
        x = self.Qfunction(x))

        # loss

h class SimpleDQN(nn.Module): def __init__(self, params):
        # init. super
        super(SimpleDQN,self).__init__()

        # save params
        self.memory_size = params.memory_size
        self.state_size = params.state_size

        # basic building blocks
        self.fc1 = nn.Linear(
                self.memory_size * (self.state_size **2), 
                self.memory_size * (self.state_size **2))
        self.fc2 = nn.Linear(
                self.memory_size * (self.state_size **2), 
                self.memory_size * (self.state_size **2))
        self.fc3 = nn.Linear(
                self.memory_size * (self.state_size **2), 
                self.state_size **2)
        self.fc4 = nn.Linear(
                self.state_size **2,
                self.memory_size)
        self.fc5 = nn.Linear(
                self.memory_size,
                5) # five actions
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = nn.Flatten(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.dropout(x)
        
        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)

        return x
        
