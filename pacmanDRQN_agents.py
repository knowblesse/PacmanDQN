from asyncio import FastChildWatcher
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
import logging
from typing import Dict
from torch.utils.tensorboard import SummaryWriter


ACTION_2_INDEX_MAPPER = {
    game.Directions.NORTH: 0,
    game.Directions.SOUTH: 1,
    game.Directions.EAST: 2,
    game.Directions.WEST: 3,
    game.Directions.STOP: 4
}
INDEX_2_ACTION_MAPPER = {v:k for k, v in ACTION_2_INDEX_MAPPER.items()}


class PacmanDRQN(game.Agent):
    def __init__(self):
        super().__init__()
        # Basic info for logging
        self.model_name = 'DRQN'
        self.exp_num = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
        self.writer = SummaryWriter(f'./runs/{self.model_name}/{self.exp_num}')
        self.logger = logging.getLogger()
        self.file_handler = logging.FileHandler(f'./logs/{self.model_name}/{self.exp_num}')
        self.logger.addHandler(self.file_handler)

        self.n_action = 5
        # hyperparameters for training
        self.batch_size = 16
        self.lstm_h_dim = 64
        self.tau = 0.01
        self.gamma = 0.9
        self.train_size = 32 # minimum number of episodes required to train the agent's model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Q estimation models, loss function, optimizer for the models.
        self.drqn_pred = DRQN(self.n_action, self.lstm_h_dim, self.lstm_h_dim).to(self.device)
        self.drqn_targ = DRQN(self.n_action, self.lstm_h_dim, self.lstm_h_dim).to(self.device)
        self.drqn_targ.load_state_dict(self.drqn_pred.state_dict())

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.drqn_pred.parameters(), lr=1e-3)

        # variables for epsilon greedy, the algorithm for selecting action.
        self.algo = epsilon_greedy_with_availablilty
        self.epsilon = 0.5
        self.eps_end = 0.001
        self.eps_decay = 0.95

        # configurations for replay memory
        self.max_epi_num = 500
        self.max_epi_len = 500
        self.lookup_step = 100
        self.random_update = True

        self.episode_record = EpisodeBuffer()
        self.memory = ReplayMemory(random_update=self.random_update,
                                   max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len,
                                   batch_size=self.batch_size,
                                   lookup_step=self.lookup_step)

        # epsiode variables
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.
        self.terminal = False
        self.reward_epi = 0.

        # cumulative statistics
        self.num_train = 0
        self.num_done_epi = 0
        self.num_won = 0

    def registerInitialState(self, state):
        # Reset epsiode information
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.
        self.terminal = False
        self.reward_epi = 0.

        self.episode_record = EpisodeBuffer()

    def observation_step(self, state):
        if self.last_action is not None:
            last_observe = util.getState(self.last_state, self.visualRange, as_numpy_arr=True)
            curr_observe = util.getState(state, self.visualRange, as_numpy_arr=True)
            print(last_observe.shape)

            last_food = self.last_state.getFoodLevel()
            curr_food= state.getFoodLevel()
            diff_food = curr_food - last_food

            if state.isLose() and curr_food > 0:
                # Our pacman got eaten by ghost
                self.last_reward = -500.
            elif state.isLose() and curr_food <= 0:
                # Died by starvation
                self.last_reward = -100.
            elif diff_food > 0:
                # Ate food
                self.last_reward = 50.
            elif any(2 in _ for _ in curr_observe):
                # saw a ghost in sight
                self.last_reward = -5
            else:
                self.last_reward = -1

            if (self.terminal and state.isWin()):
                self.last_reward = 100.
            self.reward_epi += self.last_reward

            # Store this experience in episode_record, not replay memory
            self.episode_record.put([last_observe,
                                     ACTION_2_INDEX_MAPPER[self.last_action],
                                     self.last_reward,
                                     curr_observe, self.terminal])

            if len(self.memory) > self.train_size:
                self.train()
                self.num_train += 1
                print('Train')

                if (self.num_train + 1) % self.target_update_period == 0:
                    self.update_target_network()

        self.last_state = state

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        self.reward_epi += self.last_reward
        self.terminal = True
        self.observation_step(state)

        self.rememberEpisode(self.episode_record)
        self.updateEpsilon()

        self.num_won += state.isWin()

        if self.num_done_epi % 50 == 0:
            torch.save(self.drqn_pred, self.model_name)

        score = state.getScore()
        self.writer.add_scalar('Episode rewards vs episodes', self.reward_epi, self.num_done_epi)
        self.writer.add_scalar('Episode score vs episodes', score, self.num_done_epi)
        self.writer.add_scalar('Episode won vs episodes', self.num_won, self.num_done_epi)
        self.logger.info(f'Episode: {self.num_done_epi}, Score: {score}, Reward: {self.reward_epi}, IsWin: {state.isWin()}, MemoryLen: {len(self.memory)}')

        self.num_done_epi += 1

    def getAction(self, state):
        possible_actions = state.getLegalActions(0)
        possible_action_index = [ACTION_2_INDEX_MAPPER[a] for a in possible_actions]

        with torch.no_grad():
            # state to obs
            obs = util.getState(state, self.visualRange, as_numpy_arr=True)
            obs = torch.tensor(obs)
            # print(obs.size())

            h, c = self.drqn_pred.init_hidden_state(batch_size=1, training=False)
            Q, _, _ = self.drqn_pred.forward(obs.to(self.device), h.to(self.device), c.to(self.device))
        a = self.algo(Q, self.epsilon, possible_action_index)

        action_string = INDEX_2_ACTION_MAPPER[a]
        self.last_action = action_string
        return self.last_action

    def rememberEpisode(self, episode):
        self.memory.put(episode)

    def updateEpsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def train(self):
        sampled_episodes, epi_length = self.memory.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(self.batch_size):
            observations.append(sampled_episodes[i]["obs"])
            actions.append(sampled_episodes[i]["acts"])
            rewards.append(sampled_episodes[i]["rews"])
            next_observations.append(sampled_episodes[i]["next_obs"])
            dones.append(sampled_episodes[i]["done"])

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        observations = torch.FloatTensor(observations.reshape(self.batch_size, epi_length,-1)).to(self.device)
        actions = torch.LongTensor(actions.reshape(self.batch_size, epi_length,-1)).to(self.device)
        rewards = torch.FloatTensor(rewards.reshape(self.batch_size, epi_length,-1)).to(self.device)
        next_observations = torch.FloatTensor(next_observations.reshape(self.batch_size, epi_length,-1)).to(self.device)
        dones = torch.FloatTensor(dones.reshape(self.batch_size, epi_length,-1)).to(self.device)

        h_target, c_target = self.drqn_targ.init_hidden_state(batch_size=self.batch_size, training=True)
        q_target, _, _ = self.drqn_targ.forward(next_observations, h_target.to(self.device), c_target.to(self.device))
        q_target_max = q_target.max(2)[0].view(self.batch_size, epi_length,-1).detach()
        targets = rewards + self.gamma * q_target_max * dones

        h, c = self.drqn_pred.init_hidden_state(batch_size=self.batch_size, training=True)
        q_out, _, _ = self.drqn_pred(observations, h.to(self.device), c.to(self.device))
        Q = q_out.gather(2, actions)

        loss = self.loss_fn(Q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # soft update
        for target_param, local_param in \
        zip(self.drqn_targ.parameters(), self.drqn_pred.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


class DRQN(nn.Module):
    def __init__(self, n_action, lstm_i_dim, lstm_h_dim):
        super(DRQN, self).__init__()
        self.lstm_i_dim = lstm_i_dim    # input dimension of LSTM
        self.lstm_h_dim = lstm_h_dim     # output dimension of LSTM
        self.lstm_N_layer = 1   # number of layers of LSTM
        self.n_action = n_action
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=self.lstm_i_dim, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc = nn.Linear(self.lstm_h_dim, self.n_action)

    def forward(self, x, h, c):
        x = F.relu(self.conv1(x)) # (1, 3, 7, 7)  -> (1, 32, 5, 5). RGB channel should be in second.
        x = F.relu(self.conv2(x)) # (1, 32, 5, 5) -> (1, 64, 3, 3).
        x = F.relu(self.conv3(x)) # (1, 64, 3, 3) -> (1, 64, 1, 1).
        x = x.view(x.size(0), -1) # (1, 64) remaining number of data (first dimension)
        x = x.unsqueeze(1) # -> (1, 1, 64)
        x, (new_h, new_c) = self.lstm(x, (h, c)) # (1, 1, 64)
        x = self.fc(x) # (1, 1, 5)
        x = torch.flatten(x)
        return x, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.lstm_h_dim]), torch.zeros([1, batch_size, self.lstm_h_dim])
        else:
            return torch.zeros([1, 1, self.lstm_h_dim]), torch.zeros([1, 1, self.lstm_h_dim])


class ReplayMemory:
    """
    Replay memory for recurrent agent.
    Each episode is stored into ReplayMemory's self.memory.
    """

    def __init__(self, random_update=False, 
                       max_epi_num=1000, max_epi_len=500,
                       batch_size=1,
                       lookup_step=None):
        self.random_update = random_update # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        # if (random_update is False) and (self.batch_size > 1):
        #     sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []
        # TODO: understand this logic

        ##################### RANDOM UPDATE ############################
        if self.random_update: # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)
            
            check_flag = True # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################           
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """
    A simple numpy replay buffer. This stores {(o, a, r, o', done)}_t in one episode.
    So, one EpisodeBuffer represents one episode.
    """

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


def epsilon_greedy_with_availablilty(values, epsilon, available):
    num_actions = len(values)
    available_mask = torch.tensor([i in available for i in range(0, num_actions)])
    values[~available_mask] = float('-inf')

    if random.random() > epsilon:
        a = torch.argmax(values).item()
    else:
        a = np.random.choice(available)
    return a
