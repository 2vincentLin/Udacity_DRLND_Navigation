from model import QNetwork, DDQNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque

LR = 0.0005
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
UPDATE_EVERY = 4
TAU = 0.001
ALPHA = 1
EPI = 0.001
BETA = 1
ITA = 0.25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_no_soft_update():
    '''
    most of the code come from Udacity DRLND course, because the original dqn paper didn't mention soft update,
    so I implement it and see how it performs on solving the Udacity banana collector environment,
    it can solve but not faster enough than soft update version.

    '''

    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # QNetwork
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.learn_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.learn_step = (self.learn_step + 1) % UPDATE_EVERY
        if self.learn_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        '''
        get actions
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return np.argmax(action.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''
        when learning, using target network to get q-value, then perform gradient step on expected(local) and target network,
        then set target network = local network
        '''

        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expects = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expects, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # nn.Module.paramters() is a generator,
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data)


class Agent():
    '''
    most of the code come from Udacity DRLND course, I implemented dueling and double dqn for this agent.

    '''

    def __init__(self, state_size, action_size, seed, dueling=True, double=False):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.dueling = dueling
        self.double = double

        # QNetwork
        if self.dueling:
            self.qnetwork_local = DDQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DDQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.learn_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.learn_step = (self.learn_step + 1) % UPDATE_EVERY
        if self.learn_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        '''
        get actions
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return np.argmax(action.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''
        when learning, using target network to get q-value, then perform gradient step on expected(local) and target network,
        then set target network = local network
        '''

        states, actions, rewards, next_states, dones = experiences

        if self.double:
            q_index = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            q_targets_next = self.qnetwork_target(next_states).detach().gather(1, q_index)
        else:
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expects = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expects, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # nn.Module.paramters() is a generator,
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


class ReplayBuffer():
    '''
    most of the code come from Udacity DRLND course, instead of using name tuple, I use dict.
    '''

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(
            maxlen=buffer_size)  # when append more than maxlen, old ones will automatically dropped from deque

    def add(self, state, action, reward, next_state, done):
        # using dict for readability
        e = dict({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done})
        self.memory.append(e)

    def sample(self):
        experiences = np.random.choice(self.memory, size=self.batch_size)

        states = torch.from_numpy(np.vstack([e['state'] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e['action'] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e['reward'] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e['next_state'] for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e['done'] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class SumTree:
    '''
    Most of the code come from https://github.com/jaara/AI-blog/blob/master/SumTree.py.
    However, this implementation seems having bug, sometimes it'll return empty data (haven't written yet),
    so I add a max_write to indicate max dataIdx that has data, and if self._retrieve() returns index > max_write,
    return max_write instead.

    '''
    write = 0
    max_write = 0  # it records the last self.data id that has data

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.max_write = max(self.max_write, self.write)
        self.update(idx, p)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        # for some case that algorithm retrieves dataIdx > self.max_write
        if dataIdx > self.max_write:
            #             print(' \n retrieve data from empty slot \n', self.data[dataIdx], '\n after revised \n', self.data[self.max_write])
            dataIdx = self.max_write
            idx = dataIdx - 1 + self.capacity

        return (idx, self.tree[idx], self.data[dataIdx])


class PriorityBuffer():
    '''
    https://arxiv.org/abs/1511.05952, modified from ReplayBuffer
    '''

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = SumTree(buffer_size)
        self.sampled_idx = None  # record sampled index, use it when update

    def _get_priority(self, error, epi=0.0):
        '''
        this function returns p = (error + epi) ** alpha
        :param error: td error
        :param epi: epsilon, small positive number. when updating it, it should be zero.
        :return: number
        '''
        return (np.abs(error) + epi) ** ALPHA

    def _get_weights(self, p, beta, N=BUFFER_SIZE):
        '''
        return importance sampling weight w_j = (N*P)**-beta / max w_i
        :param p: priority array / sum, P(i) = (p_j)**alpha / sum((p_j)**alpha)
        :param beta:
        :param N: buffer size
        :return: torch.tensor with size = [N]
        '''
        weights = (N * p) ** -beta
        max_w = np.max(weights)
        weights = weights / max_w  # dtype = object
        return torch.from_numpy(weights.astype(np.float32)).float().to(device)

    def add(self, state, action, reward, next_state, done, error=1):
        '''
        this function add transition and its priority to PER
        :param error: when adding it, setting error =1
        :return: None
        '''
        e = dict({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done})
        p = self._get_priority(error, EPI)
        self.memory.add(p, e)

    def sample(self):
        '''
        in the original PER paper, the author uses stratified sampling, that is, get batch size segments,
        then sample exactly one transition from each segment.
        :return: transition with probs and weights, weights will be used for updating gradient.
        '''

        segment = np.linspace(0, self.memory.total(), self.batch_size + 1)
        p_array = [np.random.uniform(a, b) for a, b in zip(segment[:-1], segment[1:])]

        batch = np.array([self.memory.get(p) for p in p_array])
        experiences = batch[:, 2]
        self.sampled_idx = batch[:, 0] # record the sampled id, so we know which ones to update.
        probs = batch[:, 1] / self.memory.total()
        weights = self._get_weights(probs, BETA)

        states = torch.from_numpy(np.vstack([e['state'] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e['action'] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e['reward'] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e['next_state'] for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e['done'] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, probs, weights)

    def update(self, errors):
        '''
        after getting td errors, call this function to update the priority, and use sample_idx to update sampled data
        :param errors: td errors
        :return: None
        '''
        for i in range(len(errors)):
            p = self._get_priority(errors[i])
            self.memory.update(self.sampled_idx[i], p)

    def __len__(self):
        return len(self.memory.data)


class PERAgent():
    '''
    modified from Agent()
    '''

    def __init__(self, state_size, action_size, seed, dueling=True, double=False):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.dueling = dueling
        self.double = double

        # QNetwork
        if self.dueling:
            self.qnetwork_local = DDQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DDQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # replay memory
        self.memory = PriorityBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.learn_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.learn_step = (self.learn_step + 1) % UPDATE_EVERY
        if self.learn_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        '''
        get actions
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return np.argmax(action.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''
        the most difficult part of the PERAgent, the paper says we need to update the model weight using
        model weight += ITA * importance sampling weight * td errors * gradient, after research,
        I use the fact that if grad = F.mse_loss(x, y), then c**2 grad = F.mse_loss(cx,cy) to update the model weight.

        :param experiences:
        :param gamma:
        :return: None
        '''
        states, actions, rewards, next_states, dones, probs, weights = experiences
        if self.double:
            q_index = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            q_targets_next = self.qnetwork_target(next_states).detach().gather(1, q_index)
        else:
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expects = self.qnetwork_local(states).gather(1, actions)

        errors = q_targets.detach() - q_expects.detach()
        weights.unsqueeze_(1) # w_j = (N*P)**-beta / max w_i, size from (N) to (N, 1)
        e_signs = torch.sign(errors) # because we need to (errors ** 2) ** 0.25, and will lose the sign

        # since error can be negative, so we need to square it then take 0.25 power
        q_expects = q_expects.mul((errors ** 2) ** 0.25).mul((weights ** 2) ** 0.25).mul(ITA ** 0.5).mul(e_signs)
        q_targets = q_targets.mul((errors ** 2) ** 0.25).mul((weights ** 2) ** 0.25).mul(ITA ** 0.5).mul(e_signs)

        self.memory.update(errors.detach().numpy())
        loss = F.mse_loss(q_expects, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # nn.Module.paramters() is a generator,
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

