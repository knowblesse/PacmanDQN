import numpy as np
import game
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

class PacmanSimpleDQN(game.Agent):
    def __init__(self):
        print("Initialize Pacman Agent (Simple DQN)")
        args = dict()
        self.memory_size = 10#args.memory_size # default : 10 # number of previous state as input
        self.state_size = 7#args.state_size # default : 7 (7x7 matrix)
        args['memory_size'] = self.memory_size
        args['state_size'] = self.state_size
        self.Qfunction = SimpleDQN(args)
        self.LTM = []
        self.WM_state_arr = [] # working memory
        self.WM_food_level = []
        self.WM_action = []
        self.WM_reward = []
        self.SM_state_arr = [] # state arrays fed to the Q value function
        self.train_size = 50#args.train_size # default : 50
        self.gamma = 0.95 # for reward discounting
        self.firstStep = True
        self.last_state = None
        self.last_action = None

    def initialize(self):
        print('Initialize')
        self.SM_state_arr = []

    def registerInitialState(self, state):
        # run once at the first state 
        self.firstStep = True
        self.last_state = state
        state_array = util.getState(state, self.state_size, True)
        self.SM_state_arr.append(state_array)

    def observationFunction(self, state):
        # observe the current state and remember 1) the past state, 2) the past action, 3) current state, and 4) reward

        if self.firstStep:
            self.firstStep = False
        else:
            state_array = util.getState(state, self.state_size, True)
            if (state.getFoodLevel() - self.last_state.getFoodLevel()) > 0 :
                # food eaten
                reward = 30
            elif np.any(state_array == 2):
                # ghost in sight
                reward = -5
            elif np.any(state_array == 3):
                # food in sight
                reward = 5
            else:
                reward = -1
            reward += state.getFoodLevel()/10

            # Register to Working Memory for training
            self.WM_state_arr.append(util.getState(self.last_state, self.state_size,True))
            self.WM_food_level.append(self.last_state.getFoodLevel())
            self.WM_action.append(self.last_action)
            self.WM_reward.append(reward)

            # Register to Short-term Memory for action selection
            if len(self.SM_state_arr)==self.memory_size:
                self.SM_state_arr.pop(0)
            self.SM_state_arr.append(state_array)

        self.last_state = state
        return state

    def getAction(self, state):
        s = self.SM_state_arr
        if len(s) < self.memory_size:
            for _ in range(self.memory_size - len(s)):
                s.insert(0,self.SM_state_arr[0])
        values = self.Qfunction.forward(np.array(s).flatten()).detach().numpy()

        # softmax function for choosing the next action
        dist = util.Counter()
        for a in state.getLegalActions(0):
            dist[a] = np.exp(values[self.__textAction2indexAction(a)])
        dist.normalize()
        action = util.chooseFromDistribution(dist) 
        self.last_action = action
        return action

    def __textAction2indexAction(self, action):
        if action == game.Directions.NORTH:
            return 0
        elif action == game.Directions.SOUTH:
            return 1 
        elif action == game.Directions.EAST:
            return 2 
        elif action == game.Directions.WEST:
            return 3
        elif action == game.Directions.STOP:
            return 4
        else:
            raise (BaseException('PacmanSimpleDQN : wrong legal Action'))

    def final(self, state):
        if state.isWin():
            reward = 100
        else:
            reward = -100

        self.WM_state_arr.append(util.getState(self.last_state, self.state_size,True))
        self.WM_food_level.append(self.last_state.getFoodLevel())
        self.WM_action.append(self.last_action)
        self.WM_reward.append(reward)

        # generate learning data from the whole episode
        for i in range(self.memory_size-1,len(self.WM_action)):
            s = np.array(self.WM_state_arr[i-(self.memory_size-1):i+1]).flatten()
            a = self.__textAction2indexAction(self.WM_action[i])
            q = np.sum(np.array(self.WM_reward[i:]) * np.array([self.gamma**i for i in range(len(self.WM_action) - i)]))

        # TODO : register learning data to the LTM

    def train(self):
        # generate train batch (from replay memory)
        dataset = np.random.choice(self.replay_mem, self.train_size, False)
        

class SimpleDQN(nn.Module):
    def __init__(self, params):
        # init. super
        super(SimpleDQN,self).__init__()

        # save params
        self.memory_size = params['memory_size']
        self.state_size = params['state_size']

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
        x = torch.from_numpy(x).type(torch.float32)
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
        
