#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import ns3gym
import gym
import pyro.distributions as dist

# Define the Q-network
class QNetwork(nn.Module):
    """ This class defines a neural network that takes 
    the state of the environment as input and outputs the Q-values 
    for each possible action"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    @classmethod
    def forward(cls, x):
        x = torch.relu(cls.fc1(x))
        x = cls.fc2(x)
        return x

# Define the Bayesian Deep Q-Learning agent
class BDQAgent:
    """ This class defines the agent that will be trained using the Bayesian Deep Q-Learning algorithm
        It also initializes the Q-network and optimizer, defines the prior distribution over the Q-network weights
          and creates a replay buffer for storing experiences.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.kl_divs = []
        self.episode_rewards = []
        self.mean_rewards = []
        self.losses = []
        self.weights = []
        
        # Initialize the Q-network and optimizer
        self.q_network = QNetwork(self.state_dim, self.hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Define the prior distribution over the Q-network weights
        self.prior_mean = torch.zeros(self.q_network.fc1.weight.shape)
        self.prior_scale = torch.ones(self.q_network.fc1.weight.shape) * 0.1
        self.prior_dist = dist.Normal(self.prior_mean, self.prior_scale).independent(2)
        
        # Define the replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Define the plot data
        self.episode_rewards = []
        self.mean_rewards = []
    @classmethod   
    def update(cls, state, action, reward, next_state, done):
        """ This function is called during each step of training and performs the following steps:
            *Adds the current experience to the replay buffer.
            *Samples a set of Q-network weights from the posterior distribution over the weights.
            *Computes the loss using the Bellman equation and mean squared error.
            *Computes the KL divergence between the posterior and prior distributions over the weights.
            *Computes the ELBO and the loss using it.
            *Updates the Q-network weights using the ELBO loss.
            """
        # Add the experience to the replay buffer
        cls.replay_buffer.append((state, action, reward, next_state, done))
        
        # Sample a set of Q-network weights from the posterior distribution
        posterior_dist = cls.get_posterior_dist()
        weights = posterior_dist.sample()
        
        # Compute the loss
        batch = random.sample(cls.replay_buffer, min(len(cls.replay_buffer), 32))
        loss = 0
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_q_values = cls.q_network(next_state)
                next_max_q_value, _ = torch.max(next_q_values, dim=1)
                target += cls.gamma * next_max_q_value.item()
            q_values = cls.q_network(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            loss += (q_value - target)**2
        loss /= len(batch)
        
        # Store the loss
        cls.losses.append(loss.item()) # add the loss to the list
        
        # Compute the KL divergence between the prior and posterior distributions
        kl_div = cls.compute_kl_div(posterior_dist, cls.prior_dist)
        cls.kl_divs.append(kl_div.item())        
        # Compute the ELBO and the loss
        elbo = -loss.sum() + kl_div.sum()
        loss = elbo / len(batch)
        
        # Update the Q-network weights using the ELBO loss
        cls.optimizer.zero_grad()
        loss.backward()
        cls.optimizer.step()
    @classmethod
    def compute_kl_div(cls, posterior_dist, prior_dist):
        """This function computes the KL divergence between two distributions"""
       
        posterior = posterior_dist.base_dist
        prior = prior_dist.base_dist
        kl_div = torch.distributions.kl.kl_divergence(posterior, prior).sum()
        return kl_div
    @classmethod
    def get_posterior_dist(cls):
        """This function defines the posterior distribution over the Q-network weights"""
        
        posterior_mean = cls.q_network.fc1.weight
        posterior_scale = torch.ones(cls.q_network.fc1.weight.shape) * 0.1
        posterior_dist = dist.Normal(posterior_mean, posterior_scale).independent(2)
        return posterior_dist
    @classmethod
    def act(cls, state):
        """function chooses an action greedily based on the Q-values computed by the Q-network."""
        
        with torch.no_grad():
            q_values = cls.q_network(state)
            action = torch.argmax(q_values, dim=1).item()
        return action
    @classmethod
    def train(cls, env, num_episodes):
        """ This function trains the agent using the Bayesian Deep Q-Learning algorithm."""
        
        for i in range(num_episodes):
            # Reset the environment
            observation = env.reset()
            state = torch.FloatTensor(observation).unsqueeze(0)
            done = False
            episode_reward = 0
            
            # Play one episode
            while not done:
                # Choose an action and observe the next state and reward
                action = cls.act(state)
                observation, reward, done, info = env.step(action)
                next_state = torch.FloatTensor(observation).unsqueeze(0)
                action = torch.LongTensor([action])
                reward = torch.FloatTensor([reward])
                
                # Update the agent
                cls.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward.item()
                
            # Update the plot data
            cls.episode_rewards.append(episode_reward)
            if len(cls.episode_rewards) >= 10:
                mean_reward = np.mean(cls.episode_rewards[-10:])
                cls.mean_rewards.append(mean_reward)
            
            # Print the episode number and reward
            print("Episode %d, Reward: %.2f" % (i+1, episode_reward))
            
        # Plot the results
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        ax1.plot(cls.episode_rewards, label='Episode Reward')
        ax1.plot(cls.mean_rewards, label='Mean Reward (last 10 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.figure.savefig("Reward_BDRL.pdf")
        
        # Plot the KL divergence
        ax2.plot(agent.kl_divs)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('KL Divergence')
        ax2.figure.savefig("BDRL_KL_Divergence.pdf")
        

        # Plot the losses
        ax3.plot(agent.losses)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.figure.savefig("loss_BDRL.pdf")
        

        # Plot the Q-network weights for each action
        for i in range(agent.action_dim):
           ax4.plot(agent.q_network.fc1.weight[:, i].detach().numpy(), label='Action {}'.format(i))
        ax4.set_xlabel('Input Dimension')
        ax4.set_ylabel('Weight Value')
        ax4.legend()
        ax4.figure.savefig("BDRL_weights.pdf")
        

        # Plot the mean rewards for each group of 10 episodes
        mean_rewards_10 = [np.mean(agent.episode_rewards[i:i+10]) for i in range(0, len(agent.episode_rewards), 10)]
        ax5.plot(mean_rewards_10)
        ax5.set_xlabel('Episode Group')
        ax5.set_ylabel('Mean Reward')
        ax5.figure.savefig("Mean_Reward_BDRL.pdf")
        

        # Plot the episode rewards over time
        ax6.plot(agent.episode_rewards)
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Reward')
        ax6.figure.savefig("Episode_Reward_BDRL.pdf")
        
	
# Create the environment and the agent
env = gym.make('ns3-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 32
gamma = 0.99
lr = 0.001
agent = BDQAgent(state_dim, action_dim, hidden_dim, gamma, lr)

# Train the agent
num_episodes = 500
agent.train(env, num_episodes)

