#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alperenadatepe
"""

import numpy as np
import matplotlib.pyplot as plt

class KArmedBanditTestBed():
    def __init__(self, num_of_bandits, mean, variance, num_of_time_steps, num_of_runs, epsilon_values):
        self.num_of_bandits = num_of_bandits
        self.mean = mean
        self.variance = variance
        self.num_of_time_steps = num_of_time_steps
        self.num_of_runs = num_of_runs
        self.epsilon_values = epsilon_values
        
    def _create_initials(self):
        self.true_q_values = np.random.normal(self.mean, self.variance, self.num_of_bandits)
        self.estimated_q_values = np.zeros((self.num_of_bandits))
            
    def choose_action_by_epsilon_greedy(self, epsilon):
        action = None
        random_value = np.random.uniform(0, 1)
        
        if random_value < epsilon:
            action = np.random.choice(self.num_of_bandits)
        else:
            action = np.argmax(self.estimated_q_values)
        
        return action
    
    def obtain_reward(self, chosen_action):
        mean_value_of_action = self.true_q_values[chosen_action]
        
        reward = np.random.normal(mean_value_of_action, self.variance)
        
        return reward
        
    def update_estimated_q_values(self, reward_list, action_list, time_step):
        for action in range(0, self.num_of_bandits):
            
            reward_sum_for_action = 0
            action_taken_number = 0
            
            for i in range(0, time_step):
                if action_list[i] == action:
                    reward_sum_for_action += reward_list[i]
                    action_taken_number += 1
            
            if action_taken_number != 0:    
                self.estimated_q_values[action] = reward_sum_for_action / action_taken_number

    
    def average_reward_experimentation(self):
        plt.figure("Average Reward Experimentation")
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        
        for epsilon in self.epsilon_values:
            average_reward_distribution = np.zeros((self.num_of_time_steps))
            
            for run in range(0, self.num_of_runs):
                self._create_initials()
                
                reward_list = []
                action_list = []
                
                for time_step in range(0, self.num_of_time_steps):
                    chosen_action = self.choose_action_by_epsilon_greedy(epsilon)
                    reward = self.obtain_reward(chosen_action)
                    
                    action_list.append(chosen_action)
                    reward_list.append(reward)
                    
                    average_reward_distribution[time_step] += reward
                    
                    self.update_estimated_q_values(reward_list, action_list, time_step)
            
            average_reward_distribution /= self.num_of_runs
            plt.plot(average_reward_distribution, label=f"{epsilon} Epsilon")
        
        plt.legend()
        
if __name__ == "__main__":
    num_of_bandits = 10
    mean = 0
    variance = 1
    num_of_time_steps = 500
    num_of_runs = 1000
    epsilon_values = [0.1, 0.01, 0.0]
                     
    k_armed_bandits_test_bed = KArmedBanditTestBed(num_of_bandits, mean, variance, num_of_time_steps, num_of_runs, epsilon_values)
    k_armed_bandits_test_bed.average_reward_experimentation()