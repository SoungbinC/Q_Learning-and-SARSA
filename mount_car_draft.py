# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:04:34 2019

@author: jg
"""
import gym

import numpy as np

print(np.__version__)


# Create the environment with proper render_mode
env = gym.make("MountainCar-v0", render_mode="human")
MAX_NUM_EPISODES = 500
GOAL_POSITION = 0.5

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = env.reset()
    total_reward = 0.0  # To keep track of the total reward obtained in each episode
    step = 0
    while not done:
        env.render()
        action = env.action_space.sample()  # Sample random action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )  # Updated to handle new API
        total_reward += reward
        step += 1
        obs = next_state

        # Check if the car has reached the goal position
        if next_state[0] >= GOAL_POSITION:
            print(
                f"Episode #{episode} succeeded in reaching the goal in {step} steps with total_reward={total_reward}"
            )
            break

        # Update the condition for done
        done = terminated or truncated

    if not next_state[0] >= GOAL_POSITION:
        print(
            "\n Episode #{} ended in {} steps. total_reward={}".format(
                episode, step + 1, total_reward
            )
        )
env.close()
