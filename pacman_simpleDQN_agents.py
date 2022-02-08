import numpy as np
import game
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import random
import datetime

SAV = namedtuple('SAV', ('state', 'action', 'vaule'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(SAV(*args))
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    def __len__(self):
        return len(self.memory)

class PacmanSimpleDQN(game.Agent):
    def __init__(self):
        print("Initializing the Pacman Agent (Simple DQN)")
        self.model_name = datetime.datetime.now().strftime('Model_%H:%M:%S.md')
        # check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parameters
        self.memory_size = 10#args.memory_size # default : 10 # number of previous state as input
        self.train_size = 100#args.train_size # default : 50
        self.gamma = 0.99 # for reward discounting
        self.state_size = 7#args.state_size # default : 7 (7x7 matrix)
        args = dict()
        args['memory_size'] = self.memory_size
        args['state_size'] = self.state_size
        args['device'] = self.device
        
        # Brain
        self.Qfunction = SimpleDQN(args).to(self.device)
        self.optimizer = torch.optim.Adagrad(self.Qfunction.parameters(),lr=0.005)
        self.score = deque([],maxlen=100)
        self.numTrain = 0

        # Memory
        self.LTM = ReplayMemory(5000) # replay memory for learning
        self.bestScore = 0

        # Working Memory # after each episode, save it into the LTM
        self.WM = {'state': [], 'action':[], 'reward':[]}

        # Short-term Memory
        self.SM_state_arr = [] # state arrays fed to the Q value function

        self.firstStep = True
        self.last_state = None
        self.last_action = None

    def initialize(self):
        # erase memory except the LTM
        self.SM_state_arr = []
        self.WM = {'state': [], 'action':[], 'reward':[]}
        self.firstStep = True
        self.last_state = None
        self.last_action = None

    def registerInitialState(self, state):
        # run once at the first state 
        self.initialize()
        self.firstStep = True
        state_array = util.getState(state, self.state_size)
        self.SM_state_arr.append(state_array)

    def observationFunction(self, state):
        # observe the current state and remember 1) the past state, 2) the past action, 3) current state, and 4) reward
        if self.firstStep:
            self.firstStep = False
        else:
            state_array = util.getState(state, self.state_size)
            state_array_np = np.array(state_array)
            if (state.getFoodLevel() - self.last_state.getFoodLevel()) > 0 :
                # food eaten
                reward = 50 * self.last_state.getFoodLevel() / 10
                print('food eaten')
            elif np.any(state_array_np == 2):
                # ghost in sight
                reward = -5
            elif np.any(state_array_np == 3):
                # food in sight
                points = np.where(state_array_np == 3)
                nearFoodScore = 0
                for x,y in zip(*points):
                    nfs = abs((self.state_size-1)/2 - x) + abs((self.state_size-1)/2 - y)
                    if nfs > nearFoodScore:
                        nearFoodScore = nfs
                reward = nearFoodScore
            else:
                reward = -1

            # Register to Working Memory for training
            self.WM['state'].append(util.getState(self.last_state, self.state_size))
            self.WM['action'].append(self.last_action)
            self.WM['reward'].append(reward)

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
        values = self.Qfunction.forward(torch.tensor(s, dtype=torch.float32)).detach().to("cpu").numpy()

        # softmax function for choosing the next action
        dist = dict()
        if np.random.rand()<0.05:
            for a in state.getLegalActions(0):
                dist[a] = 1
        else:
            for a in state.getLegalActions(0):
                dist[a] = np.exp(values[self.__textAction2indexAction(a)])

        total = 0
        for i in dist:
            total += dist[i]
        choice = np.random.rand() * total
        cumval = 0
        for i in dist:
            cumval += dist[i]
            if choice < cumval:
                break
        action = i
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
            reward = -30

        self.WM['state'].append(util.getState(self.last_state, self.state_size))
        self.WM['action'].append(self.last_action)
        self.WM['reward'].append(reward)

        # generate learning data from the whole episode
        if state.getScore() >= np.mean(self.score):
            for i in range(self.memory_size-1,len(self.WM['action'])):
                self.LTM.push(
                    self.WM['state'][i-(self.memory_size-1):i+1],
                    self.__textAction2indexAction(self.WM['action'][i]),
                    sum([r*d for r, d in zip(self.WM['reward'][i:],[self.gamma**i for i in range(len(self.WM['action']) - i)])]))
            self.bestScore = state.getScore()
            print("memorized")
        if len(self.LTM) > 2000:
            self.train()
            print("trained")
        self.score.append(state.getScore())
        if self.numTrain % 50 == 0:
            torch.save(self.Qfunction, self.model_name)
            print("saved")
        print(f'{self.numTrain} : {np.mean(self.score)}')
        self.numTrain = self.numTrain + 1

    def train(self):
        # generate train batch (from replay memory)
        sample = self.LTM.sample(self.train_size)
        sample = SAV(*zip(*sample))

        X = torch.tensor(sample.state, dtype=torch.float32)

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(self.Qfunction.forward(X).gather(1,torch.tensor(sample.action,device=self.device).unsqueeze(1)), torch.tensor(sample.vaule, device=self.device).unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class SimpleDQN(nn.Module):
    def __init__(self, params):
        # init. super
        super(SimpleDQN,self).__init__()

        # save params
        self.memory_size = params['memory_size']
        self.state_size = params['state_size']
        self.device = params['device']

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
        self.bn = nn.BatchNorm1d(self.state_size**2*self.memory_size)
        self.dropout = nn.Dropout(0.05)



    def forward(self, x):
        x = x.to(self.device)
        if x.dim() == 4:
            x = self.bn(x.flatten(1))
        else:
            x = x.flatten()

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)

        return x

