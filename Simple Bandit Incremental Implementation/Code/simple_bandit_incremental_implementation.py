#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alperenadatepe
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleBandit():
    def __init__(self, num_of_bandits, mean, variance, num_of_time_steps, epsilon_value):
        self.num_of_bandits = num_of_bandits
        self.mean = mean
        self.variance = variance
        self.num_of_time_steps = num_of_time_steps
        self.epsilon_value = epsilon_value
        
        self._create_initials()
        
    def _create_initials(self):
        self.true_q_values = np.random.normal(self.mean, self.variance, self.num_of_bandits)
        self.estimated_q_values = np.zeros((self.num_of_bandits))
        self.number_of_action_taken = np.zeros((self.num_of_bandits))

    def choose_action_by_epsilon_greedy(self):
        action = None
        random_value = np.random.uniform(0, 1)
        
        if random_value < self.epsilon_value:
            action = np.random.choice(self.num_of_bandits)
        else:
            action = np.argmax(self.estimated_q_values)
        
        return action
        
    def provide_reward(self, chosen_action):
        mean_value_of_the_action = self.true_q_values[chosen_action]
        
        return np.random.normal(mean_value_of_the_action, 1)
    
    def simple_bandit_play(self):

        for time_step in range(0, self.num_of_time_steps):
            chosen_action = self.choose_action_by_epsilon_greedy()
            reward = self.provide_reward(chosen_action)
            
            self.number_of_action_taken[chosen_action] += 1
            
            step_size = 1 / self.number_of_action_taken[chosen_action]
        
            self.estimated_q_values[chosen_action] = self.estimated_q_values[chosen_action] + step_size * (reward - self.estimated_q_values[chosen_action])
            
    def plot_values(self):
        plt.title("True vs. Estimated Values")
        plt.plot(self.true_q_values, label="True Values")
        plt.plot(self.estimated_q_values, label="Estimated Values")
        plt.xlabel("Bandits")
        plt.ylabel("Values of each bandits")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    num_of_bandits = 10
    mean = 0
    variance = 1
    num_of_time_steps = 1000
    epsilon_value = 0.1
    
    simple_bandit = SimpleBandit(num_of_bandits, mean, variance, num_of_time_steps, epsilon_value)
    simple_bandit.simple_bandit_play()
    simple_bandit.plot_values()