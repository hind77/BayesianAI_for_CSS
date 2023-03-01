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
print("this is the action space n",action_space)

class BayesianThompsonPolicy:
    """Define the Bayesian Policy with Thompson Sampling"""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.counts = np.zeros(action_space.n, dtype=np.int)
        self.values = np.zeros(action_space.n, dtype=np.float)
        self.alpha = np.ones(env.action_space.n, dtype=np.float)
        self.beta = np.ones(env.action_space.n, dtype=np.float)
    @classmethod
    def sample(cls, observation):
        theta = np.random.beta(cls.alpha, cls.beta)
        action = np.argmax(theta)
        return action
    @classmethod
    def update(cls, observation, action, reward):
        cls.counts[action] += 1
        n = cls.counts[action]
        value = cls.values[action]
        cls.values[action] = ((n - 1) / n) * value + (1 / n) * reward
        cls.alpha[action] += reward
        cls.beta[action] += (1 - reward)

# Define the main algorithm loop
num_episodes = 100
total_rewards = np.zeros(num_episodes)
avg_rewards = np.zeros(num_episodes)
policy = BayesianThompsonPolicy(env.observation_space, env.action_space)

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
plt.title('BRL with Thompson Sampling')
plt.savefig("BRL_Thompson_Sampling_total_reward.pdf")
plt.show()


#plot the average reward per episode
plt.plot(avg_rewards)
plt.title('Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("BRL_Thompson_Sampling_average_reward.pdf")
plt.show()
#the distribution of rewards
plt.hist(total_rewards, bins=50)
plt.title('Distribution of Rewards')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.savefig("BRL_Thompson_Sampling_distribution_reward.pdf")
plt.show()

#Plot the Q-values
m_states = env.observation_space.n
m_actions = env.action_space.n
q_values = np.zeros((num_states, num_actions))

for state in range(num_states):
    for action in range(num_actions):
        q_values[state][action] = np.max(Q[state][action])

plt.imshow(q_values.T, cmap='hot', interpolation='nearest')
plt.title('Q-Values for each Action in each State')
plt.xlabel('State')
plt.ylabel('Action')
plt.colorbar()
plt.savefig("BRL_Thompson_Sampling_Q_values.pdf")
plt.show()