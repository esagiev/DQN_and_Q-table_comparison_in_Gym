import gym
import numpy as np
import matplotlib.pyplot as plt


def act_gen(place, speed):
    ''' Generation of action according to
    epsilon-greedy policy. '''
    if np.random.rand(1) <= explore_rate:
        return np.random.randint(0,3)
    else:
        return np.argmax(Q[place, speed])

def change(state):
    ''' Discretization of state values. '''
    place = int((state[0]+1.2)/0.06)
    speed = int((state[1]+0.07)/0.07*10)
    return (place, speed)

def evol(Q):
    ''' Evaluation of current policy. '''
    state = env.reset()
    episode_reward = 0.0
    while True:
        state = change(state)
        action = np.argmax(Q[state[0], state[1]])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done: break
    return episode_reward


# Q-learning with original MDP. Vanilla case.

env = gym.make('MountainCar-v0')
Q = np.zeros([31,21,3]) # Initiate Q-table
total_reward = [] # Collect episode reward
train_reward = [] # Track training
er_track = []
lr_track = []

episodes = 1000000
gamma = 1.0 # In [0.75, 1.0]

lr = 0.9 # Learning rate (alpha). In [0.2, 0.3]
lr_decay = 0.9999
lr_min = 0.01

explore_rate = 0.5
er_decay = 0.9999
er_min = 0.05 # In [0.01, 0.03]

# Main loop
for episode in range(episodes):
    state_cur = change(env.reset())
    action_cur = np.random.randint(0,3)
    episode_reward = 0.0
    
    while True:
        # Current Q-value
        Q_cur = Q[state_cur[0], state_cur[1], action_cur]
        # New action ans state
        action_next = act_gen(state_cur[0], state_cur[1])
        state_next, reward, done, _ = env.step(action_next)
        state_next = change(state_next)
        episode_reward += reward
        # Get next Q-value
        Q_max = np.max(Q[state_next[0], state_next[1]])
        # Update current Q-value
        if done:
            Q[state_cur[0], state_cur[1], action_cur] = reward
            break
        else:
            Q[state_cur[0], state_cur[1], action_cur] = \
                      (1-lr)*Q_cur + lr*(reward + gamma*Q_max)
        # Update variables for new loop
        state_cur = state_next
        action_cur = action_next
    
    # Update of hyperparameters and tracking
    er_track.append(explore_rate)
    lr_track.append(lr)
    #if explore_rate >= er_min:
    #    explore_rate *= er_decay
    #if lr >= lr_min:
    #    lr *= lr_decay
    if episode % 2000 == 0:
        total_reward.append(evol(Q))
        train_reward.append(episode_reward)