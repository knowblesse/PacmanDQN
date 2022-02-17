import numpy as np
import game
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import random
import datetime
import matplotlib.pyplot as plt

Event = namedtuple('Event', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Event(*args))

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class PacmanDDQN(game.Agent):
    """
    * How the pacman environment calls agent functions
        1. Load the Agents by calling
            __init__(self)
        2. Start a new game(episode)
        3. The initial game state is given by calling
            registerInitialState(self, state)
        4. Now the agent can observe the game state by calling
            observationFunction(self, state)
            (The agent can observe the game state from registerInitialState(), but observing through this function makes the code simple.)
        5. The agent select action by calling
            getAction(self, state)
        6. The pacman environment accepts this action and generate a new State.
        7. If the game is not over, go back to the Step 4. else, continue
        8. At the end of the game, the agent calls
            final(self, state)
        9. Go back to the Step 2 and start a new game
    """
    """
    Implemented Double DQN by separating the policy_net and the target_net.
    The policy_net is used to 
        1) select action by calculating values of each actions
        2) generating expected value for the Q function update. 
    The target_net is only used to calculate the true value for the Q function update.
    Only the policy_net is directly trained.
    The target_net is updated by "soft_update" manners.
        self.tau value is used to calculate the weighted sum of two network, 
        and this value is directed fed to the target_net
    """

    def __init__(self):
        super().__init__()
        # Constant Variables
        self.model_name = datetime.datetime.now().strftime('Model_%H:%M:%S.md')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualRange = 7 #(7x7 matrix)
        self.inputStackSize = 10 # number of previous state as input
        
        # Hyper Parameters
        self._memorySize = 10000
        self._trainSize = 512 # batch size for training
        self._gamma = 0.999 # reward discounting
        self._initialEpsilon = 1
        self._epsilonDenominator = 400 # epsilon = _initialEpsilon / (1 + numEpisode / _epsilonDenominator)
        self._initialLr = 0.001
        self._tau = 0.01
        
        # Q function Variables
        args = dict()
        args['memory_size'] = self.inputStackSize
        args['state_size'] = self.visualRange
        args['device'] = self.device
        self.policy_net = SimpleDQN(args).to(self.device)
        self.target_net = SimpleDQN(args).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy the policy_net
        self.target_net.eval() # set as evaluation mode.
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(),lr=self._initialLr)
        self.lr = self._initialLr

        # Agent Variables
        self.epsilon = self._initialEpsilon
        self.numStep = 1
        self.lastState = None
        self.last_action = None
        self.lastInputStack = None
        self.inputStack = deque([], maxlen=self.inputStackSize)
        self.Memory = ReplayMemory(self._memorySize) # replay memory for learning

        # Agent Metrics
        self.numTraining = 0
        self.numEpisode = 0
        self.score = deque([],maxlen=100)
        self.scoreHistory = []
        self.bestScore = 0

    def registerInitialState(self, state):
        # runs at the beginning of the new episode
        self.numStep = 1
        self.lastState = None
        self.last_action = None
        self.inputStack.clear()
        self.lastInputStack = None

    def observationFunction(self, state):
        # Observe the current state
        state_array = util.getState(state, self.visualRange)

        # for the first step, append inputStack and continue
        if self.numStep == 1:
            self.inputStack.append(state_array)
        # for other step, calculate the reward, save to Replay Memory
        else:
            # Calculate the reward
            if (state.getFoodLevel() - self.lastState.getFoodLevel()) > 0 :
                # food eaten
                reward = 100
            elif any(2 in _ for _ in state_array):
                # ghost in sight
                reward = -5
            else:
                reward = -1

            # Save previous inputStack array
            self.lastInputStack = self.__getInputStack()

            # Append current state array to the inputStack
            self.inputStack.append(state_array)

            # Save to the Replay Memory
            self.Memory.push(
                    self.lastInputStack,
                    self.__textAction2indexAction(self.last_action),
                    reward,
                    self.__getInputStack())

            # If there are sufficient amount of steps stored in the Memory, execute train()
            if len(self.Memory) > self._trainSize:
                self.train()
                self.numTraining += 1

            # Copy the policy_net weight to the target_net weight
            for target_net_param, policy_net_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_net_param.data.copy_(
                    self._tau * policy_net_param.data + (1.0 - self._tau) * target_net_param.data)

        self.lastState = state
        return state

    def getAction(self, state):
        # Select Actions
        possibleActions = state.getLegalActions(0)
        if np.random.rand() < self.epsilon: # Go Random
            action = np.random.choice(possibleActions)
        else: # Go Greedy
            # Calculate values for each actions
            self.policy_net.eval()  # set the policy net to evaluation mode
            with torch.no_grad():
                values = self.policy_net.forward(torch.tensor(self.__getInputStack(), dtype=torch.float32)).detach().to(
                    "cpu").numpy()
            self.policy_net.train()  # set the policy net back into training mode

            maxAction = dict(action=None, value=float("-inf"))
            for a in possibleActions:
                if values[self.__textAction2indexAction(a)] > maxAction['value']:
                    maxAction['action'] = a
                    maxAction['value'] = values[self.__textAction2indexAction(a)]
            action = maxAction['action']
        self.last_action = action
        self.numStep += 1
        return action

    def final(self, state):
        # Calculate the reward
        if state.isWin():
            reward = 10
        else:
            reward = -10

        # Save previous inputStack array
        self.lastInputStack = self.__getInputStack()

        # Save to the Replay Memory
        self.Memory.push(
                self.lastInputStack,
                self.__textAction2indexAction(self.last_action),
                reward,
                None)

        # Calculate epsilon
        self.epsilon = self._initialEpsilon / (1 + self.numEpisode / self._epsilonDenominator)

        # save the model
        if self.numEpisode % 50 == 0:
            torch.save(self.policy_net, self.model_name)

        score = state.getScore()
        # Save the Best score
        if self.bestScore < score:
            self.bestScore = score

        self.numEpisode += 1
        self.score.append(score)
        self.scoreHistory.append(score)
        print(f'Episode: {self.numEpisode:4d} | Score: {score} | Score(moving): {np.mean(self.score):5.2f} | numTrain: {self.numTraining:6d} | epsilon: {self.epsilon:.2f} | memory: {len(self.Memory)}')
        plt.clf()
        plt.plot(self.scoreHistory)
        plt.draw()
        plt.pause(0.01)

    def train(self):
        # generate train batch (from replay memory)
        sample = self.Memory.sample(self._trainSize)
        sample = Event(*zip(*sample))

        S = torch.tensor(sample.state, dtype=torch.float32, device=self.device)
        A = torch.tensor(sample.action, device=self.device).unsqueeze(1)
        R = torch.tensor(sample.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        N = sample.next_state

        # check through all steps in sample and if the next state is none, set value to zero.
        # if next state exist, use the target_net to compute the maximum value of the next state
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, sample.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in N if s is not None], device=self.device, dtype=torch.float32)

        next_state_values = torch.zeros(self._trainSize, device=self.device)
        next_state_values[non_final_mask] = torch.max(self.target_net.forward(non_final_next_states).detach(), 1).values

        self.optimizer.zero_grad()
        loss = F.mse_loss(self.policy_net.forward(S).gather(1, A), R + self._gamma * next_state_values.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.7) # limit the graidient change
        self.optimizer.step()

    def __getInputStack(self):
        # return inputStack. If the inputStack is not full, populate with the oldest state data
        iS = self.inputStack
        if len(iS) < self.inputStackSize:
            for _ in range(self.inputStackSize - len(iS)):
                iS.appendleft(self.inputStack[0])
        return iS

    def __textAction2indexAction(self, action):
        # Convert str format action into the int
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


class SimpleDQN(nn.Module):
    def __init__(self, params):
        # init. super
        super(SimpleDQN,self).__init__()

        # save params
        self.inputStackSize = params['memory_size']
        self.visualRange = params['state_size']
        self.device = params['device']

        # basic building blocks
        self.fc1 = nn.Linear(
                self.inputStackSize * (self.visualRange **2), 
                self.inputStackSize * (self.visualRange **2))
        self.fc2 = nn.Linear(
                self.inputStackSize * (self.visualRange **2), 
                self.inputStackSize * (self.visualRange **2))
        self.fc3 = nn.Linear(
                self.inputStackSize * (self.visualRange **2), 
                self.visualRange **2)
        self.fc4 = nn.Linear(
                self.visualRange **2,
                self.inputStackSize)
        self.fc5 = nn.Linear(
                self.inputStackSize,
                5) # five actions
        self.bn = nn.BatchNorm1d(self.visualRange**2*self.inputStackSize)
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

        #x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)

        return x

