#!/usr/bin/env python3
import gym
import ns3gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import seaborn as sns

# Define the NS-3 environment
env = gym.make('ns3-v0')
observation_space = env.observation_space.shape
action_space = env.action_space.n
print("this is the action space n", action_space)

# Define the Bayesian Policy with Upper Confidence Bound (UCB)
class BayesianUCBPolicy:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.counts = np.zeros(action_space.n, dtype=np.int)
        self.values = np.zeros(action_space.n, dtype=np.float)
        self.upper_bounds = np.zeros(action_space.n, dtype=np.float)
        self.t = 0
    @classmethod
    def sample(cls, observation):
        cls.t += 1
        for a in range(cls.action_space.n):
            if cls.counts[a] == 0:
                return a
        ucb_values = cls.values + np.sqrt(2 * np.log(cls.t) / cls.counts)
        action = np.argmax(ucb_values)
        return action
    @classmethod
    def update(cls, observation, action, reward):
        cls.counts[action] += 1
        n = cls.counts[action]
        value = cls.values[action]
        cls.values[action] = ((n - 1) / n) * value + (1 / n) * reward

# Define the main algorithm loop
num_episodes = 100
total_rewards = np.zeros(num_episodes)
avg_rewards = np.zeros(num_episodes)
policy = BayesianUCBPolicy(env.observation_space, env.action_space)

for i_episode in range(num_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = policy.sample(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        policy.update(obs, action, reward)

    total_rewards[i_episode] = ep_reward
    avg_rewards[i_episode] = np.mean(total_rewards[max(0, i_episode-10):i_episode+1])

# Plot the results
sns.lineplot(x=range(num_episodes), y=total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('BRL with UCB')
plt.savefig("BRL_UCB_total_reward.pdf")
plt.show()


# Plot the average reward per episode
plt.plot(avg_rewards)
plt.title('Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("BRL_UCB_average_reward.pdf")
plt.show()

# Plot the distribution of rewards
plt.hist(total_rewards, bins=50)
plt.title('Distribution of Rewards')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.savefig("BRL_UCB_distribution_reward.pdf")
plt.show()

# Plot the Q-values
num_states = np.prod(env.observation_space.shape)
num_actions = env.action_space.n
q_values = np.zeros((num_states, num_actions))

for state in range(num_states):
    for action in range(num_actions):
        obs = np.array([state, action])
        q_values[state][action] = policy.values[action]

plt.imshow(q_values.T, cmap='hot', interpolation='nearest')
plt.title('Q-Values for each Action in each State')
plt.xlabel('State')
plt.ylabel('Action')
plt.colorbar()
plt.savefig("BRL_UCB_Q_values.pdf")
plt.show()