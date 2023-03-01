#!/usr/bin/env python3
import gym
import tensorflow as tf
import tf_slim  as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env
from tensorflow.keras.layers import LeakyReLU
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Bayesian Neural Network model
def build_bnn_model(n_input, n_hidden):
    with pm.Model() as model:
        # Priors on the weights and biases
        weights_in = pm.Normal('weights_in', mu=0, sd=1, shape=(n_input, n_hidden))
        weights_out = pm.Normal('weights_out', mu=0, sd=1, shape=n_hidden)
        bias_in = pm.Normal('bias_in', mu=0, sd=1, shape=n_hidden)
        bias_out = pm.Normal('bias_out', mu=0, sd=1)

        # Input and output data
        x = pm.Data('x', np.zeros((1, n_input)))
        y = pm.Data('y', np.zeros(1))

        # Hidden layer with ReLU activation
        hidden = pm.math.maximum(pm.math.dot(x, weights_in) + bias_in, 0)

        # Output layer with linear activation
        output = pm.math.dot(hidden, weights_out) + bias_out

        # Likelihood function (Gaussian)
        y_obs = pm.Normal('y_obs', mu=output, sd=1, observed=y)

    return model

# Define the modified step function
def step(self, action):
    # Sample from the posterior predictive distribution
    x = np.array([self.state + [action]])
    with self.bnn_model:
        ppc = pm.sample_posterior_predictive(self.trace, vars=[self.y_obs], samples=100)
        y_pred = ppc['y_obs'].mean(axis=0)
    reward = y_pred[0]

    # Update the state and done flag
    self.state = self.get_obs()
    done = (self.state[0] >= self.max_time)

    return self.state, reward, done, {}

# Define the training loop
env = gym.make('ns3-v0')
env.bnn_model = build_bnn_model(env.observation_space.shape[0] + 1, 5) # add 1 for the action
with env.bnn_model:
    trace = pm.sample(10000, tune=100, cores=1)
env.trace = trace
rewards = []
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    rewards.append(total_reward)

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('BNN_Rewards.pdf')
plt.show()

# Plot the posterior distribution over the weights and biases
with env.bnn_model:
    pm.traceplot(trace, var_names=['weights_in', 'weights_out', 'bias_in', 'bias_out'])
    plt.savefig('posterior_distribution_BNN.pdf', bbox_inches='tight')
    plt.show()

# Plot the posterior predictive distribution over the output for a given input
x = np.array([[0.5, 0.1, 0.2, 0.3, 0.4, 0]]) # example input
with env.bnn_model:
    ppc = pm.sample_posterior_predictive(trace, vars=[env.y_obs], samples=100, feed_dict={env.x: x})
    y_pred = ppc['y_obs'].mean(axis=0)
sns.kdeplot(y_pred, shade=True, xlabel='Output')
plt.savefig('Posterior_predictive_distribution_BNN.pdf', bbox_inches='tight')
plt.show()

