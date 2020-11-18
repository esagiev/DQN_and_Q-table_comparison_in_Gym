import collections
import gym
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    ''' Q-network to approximate Q-function. '''
    def __init__(self):
        super(DQN, self).__init__()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.hidden = 1024
        self.l1 = nn.Linear(self.state_size, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_size, bias=False)

    def forward(self, state):
        model = nn.Sequential(
            self.l1,
            nn.Tanh(),
            self.l2
        )
        return model(state)


class Buffer():
    ''' Memory for experience replay. '''
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def rec(self, sample):
        return self.buffer.append(sample)
    
    def gen(self):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, state_next, done = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(state), np.array(action), np.array(reward), \
            np.array(state_next), np.array(done)


def exp_gen(net, buffer):
    ''' Generation of new experience for replay. '''
    state = env.reset()
    episode_reward = 0.0
    while True:
        if np.random.rand() < explore_rate:
            action = env.action_space.sample()
        else:
            Q = net(torch.tensor(state).float())
            action = torch.max(Q, -1)[1].item()
        
        state_next, reward, done, _ = env.step(action)
        buffer.rec((state, action, reward, state_next, done))
        episode_reward += reward
        state = state_next
        if done: break
    return episode_reward

def pol_eval(net):
    ''' Current policy evaluation. '''
    state = env.reset()
    episode_reward = 0.0
    while True:
        Q = net(torch.tensor(state).float())
        action = torch.max(Q, -1)[1].item()
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done: break
    return episode_reward


def train(net, net_fix, optimizer, loss_fun, buffer):
    ''' Train DQN on random batch of samples. '''
    state, action, reward, state_next, done = buffer.gen()
    
    state_v = torch.tensor(state).float()
    action_v = torch.tensor(action).long()
    reward_v = torch.tensor(reward).float()
    state_next_v = torch.tensor(state_next).float()
    done_mask = torch.ByteTensor(done)
    
    Q_bid = net(state_v).gather(1, action_v.unsqueeze(-1)).squeeze(-1)
    Q_target = net_fix(state_next_v).max(1)[0]
    Q_target[done_mask] = 0.0
    Q_target = reward_v + gamma*Q_target.detach()
    
    optimizer.zero_grad()
    loss = loss_fun(Q_bid, Q_target)
    loss.backward()
    optimizer.step()


# Parameters
episodes = 350000
buffer_size = 10000
batch_size = 50
sync_frq = 50

explore_rate = 1.0
er_decay = 0.9999
er_min = 0.05

learning_rate = 0.1
lr_min = 0.01
gamma = 0.99

# Initialization
env = gym.make('MountainCar-v0')
net = DQN()
net_fix = DQN()

buffer = Buffer()
loss_fun = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.99)
total_reward = []
train_reward = []
lr_track = []
er_track = []

# Main loop
for episode in range(episodes):
    
    # Get new experience and place into buffer
    reward = exp_gen(net, buffer)
    if len(buffer) < buffer_size:
        continue
    
    # Train model from buffer
    train(net, net_fix, optimizer, loss_fun, buffer)
    
    # Update fixed Q-target
    if episode % sync_frq == 0:
        net_fix.load_state_dict(net.state_dict())
        train_reward.append(reward) # Track training
        total_reward.append(pol_eval(net)) # Track efficiency
    
    # Tracking
    er_track.append(explore_rate)
    for param_group in optimizer.param_groups:
        lr_track.append(param_group['lr'])
    
    # Update parameters
    if explore_rate > er_min:
        explore_rate *= er_decay
    if lr_track[-1] > lr_min:
        scheduler.step()

torch.save(net.state_dict(), 'MountCar.pt')
