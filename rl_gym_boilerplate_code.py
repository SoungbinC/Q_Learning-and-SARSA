# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:39:02 2019

@author: jg
"""

#import gym
#env = gym.make("ENVIRONMENT_NAME")
#obs = env.reset() # The first arrow in the picture
## Inner loop (roll out)
#action = agent.choose_action(obs) # The second arrow in the picture
#next_state, reward, done, info = env.step(action) # The third arrow (and more)
#obs = next_state
## Repeat Inner loop (roll out)

import gym
env = gym.make("Qbert-v0")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()# Sample random action. This will be replaced by our agent's action when we start developing the agent algorithms
        next_state, reward, done, info = env.step(action) # Send the action to the environment and receive the next_state, reward and whether done or not
        obs = next_state

        if done is True:
            print("\n Episode #{} ended in {} steps.".format(episode, step+1))
            break