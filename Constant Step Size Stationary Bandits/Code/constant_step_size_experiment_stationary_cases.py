#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alperenadatepe
"""

import numpy as np
import matplotlib.pyplot as plt

class KArmedBanditTestBed():
    def __init__(self, num_of_bandits, mean, variance, num_of_time_steps, num_of_runs, epsilon_value, step_size_values):
        self.num_of_bandits = num_of_bandits
        self.mean = mean
        self.variance = variance
        self.num_of_time_steps = num_of_time_steps
        self.num_of_runs = num_of_runs
        self.epsilon_value = epsilon_value
        self.step_size_values = step_size_values
        
    def _create_initials(self):
        self.true_q_values = np.random.normal(self.mean, self.variance, self.num_of_bandits)
        self.estimated_q_values = np.zeros((self.num_of_bandits))
            
    def choose_action_by_epsilon_greedy(self):
        action = None
        random_value = np.random.uniform(0, 1)
        
        if random_value < self.epsilon_value:
            action = np.random.choice(self.num_of_bandits)
        else:
            action = np.argmax(self.estimated_q_values)
        
        return action
    
    def provide_reward(self, chosen_action):
        mean_value_of_action = self.true_q_values[chosen_action]
        
        reward = np.random.normal(mean_value_of_action, self.variance)
                
        return reward
            
    def constant_step_size_exp_stationary_cases(self):
        plt.figure("Constant Step Sizes For Stationary Cases")
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        
        for step_size in self.step_size_values:
            average_reward_distribution = np.zeros((self.num_of_time_steps))
            
            for run in range(0, self.num_of_runs):
                self._create_initials()
                
                for time_step in range(0, self.num_of_time_steps):
                    chosen_action = self.choose_action_by_epsilon_greedy()
                    reward = self.provide_reward(chosen_action)
                    
                    average_reward_distribution[time_step] += reward
                    
                    self.estimated_q_values[chosen_action] = self.estimated_q_values[chosen_action] + step_size * (reward - self.estimated_q_values[chosen_action])            
            
            average_reward_distribution /= self.num_of_runs
            plt.plot(average_reward_distribution, label=f"{step_size} Step Size")
        
        plt.legend()
        
if __name__ == "__main__":
    num_of_bandits = 10
    mean = 0
    variance = 1
    num_of_time_steps = 3000
    num_of_runs = 3000
    epsilon_value = 0.1
    step_size_values = [0.1, 0.5, 0.9]
                     
    k_armed_bandits_test_bed = KArmedBanditTestBed(num_of_bandits, mean, variance, num_of_time_steps, num_of_runs, epsilon_value, step_size_values)
    k_armed_bandits_test_bed.constant_step_size_exp_stationary_cases()