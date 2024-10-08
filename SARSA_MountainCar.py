#!/usr/bin/env/ python
"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. |Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
import gym
import numpy as np
import os
import json
import time

print(np.__version__)

# MAX_NUM_EPISODES = 500
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200  #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim

GOAL_POSITION = 0.5  # Success if the car's position reaches 0.5


class SARSA_Learner:
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def extract_obs(self, obs):
        """Extracts the observation from the given obs variable."""
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the first element if obs is a tuple
        return obs

    def discretize(self, obs):
        obs = self.extract_obs(obs)
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs, next_action):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        # Updated Q-value using SARSA
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        td_target = reward + self.gamma * self.Q[discretized_next_obs][next_action]
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error


def train(agent, env, metrics):
    best_reward = -float("inf")
    success_count = 0  # To track the number of successful episodes

    for episode in range(MAX_NUM_EPISODES):
        start_time = time.time()
        first_success_episode = None
        done = False
        obs = env.reset()
        obs = agent.extract_obs(obs)  # Extract observation if necessary

        # Select the first action
        action = agent.get_action(obs)

        total_reward = 0.0
        step = 0

        while not done:
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = agent.extract_obs(next_obs)  # Extract observation if necessary

            """
            CHanged from Q-learning to SARSA
            go to the next action on current stage
            """
            # Select next action
            next_action = agent.get_action(next_obs)

            # Update Q-value using SARSA
            agent.learn(obs, action, reward, next_obs, next_action)
            # Move to the next state and action
            obs = next_obs
            action = next_action
            total_reward += reward
            step += 1

            # Check if the goal is reached
            if next_obs[0] >= GOAL_POSITION and first_success_episode is None:
                first_success_episode = episode
                metrics["first_success_episode"] = episode

            # Check if the goal is reached
            if next_obs[0] >= GOAL_POSITION:
                metrics["success_count"] += 1
                break

        # Track end time of the episode and calculate duration
        end_time = time.time()
        episode_duration = end_time - start_time

        metrics["total_rewards"].append(total_reward)
        metrics["steps_per_episode"].append(step)
        metrics["training_time_per_episode"].append(episode_duration)
        metrics["epsilon_values"].append(agent.epsilon)
        if total_reward > best_reward:
            best_reward = total_reward

        print(
            f"Episode#: {episode} reward: {total_reward} best_reward: {best_reward} eps: {agent.epsilon}"
        )

    print(f"Number of successful episodes: {success_count}")
    # Return the trained policy
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):

    done = False
    obs = env.reset()
    obs = agent.extract_obs(obs)  # Extract observation if necessary

    total_reward = 0.0
    success = False  # To track if the goal is reached
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, truncated, info = env.step(action)
        next_obs = agent.extract_obs(next_obs)  # Extract observation if necessary
        obs = next_obs
        total_reward += reward

        # Check if the car reached the goal
        if obs[0] >= GOAL_POSITION:
            success = True

    return total_reward, success


if __name__ == "__main__":
    # Load metrics from metrics.json
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    else:
        # If metrics.json doesn't exist, initialize the metrics dictionary with separate keys
        metrics = {
            "Q_learning": {
                "total_rewards": [],
                "steps_per_episode": [],
                "success_count": 0,
                "epsilon_values": [],
                "test_success_count": 0,
                "test_failure_count": 0,
                "successful_episodes": [],
            },
            "SARSA": {
                "total_rewards": [],
                "steps_per_episode": [],
                "success_count": 0,
                "epsilon_values": [],
                "test_success_count": 0,
                "test_failure_count": 0,
                "successful_episodes": [],
            },
        }
    SARSA_metrics = metrics["SARSA"]
    # Create the training environment
    env = gym.make("MountainCar-v0")
    agent = SARSA_Learner(env)

    # Train the agent
    learned_policy = train(agent, env, SARSA_metrics)

    # Create the testing environment with "rgb_array" render mode for recording
    record_path = "./gym_record_output_sarsa"
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, record_path, episode_trigger=lambda x: True)

    # Track the results
    success_count = 0
    failure_count = 0
    successful_episodes = []

    # Test and record 10 episodes with the learned policy, tracking success
    for i in range(10):
        total_reward, success = test(agent, env, learned_policy)
        if success:
            success_count += 1
            successful_episodes.append(i)
            print(
                f"Test Episode #{i} total_reward: {total_reward} - SUCCESS (Reached the Goal!)"
            )
        else:
            failure_count += 1
            print(
                f"Test Episode #{i} total_reward: {total_reward} - FAILED (Did Not Reach the Goal)"
            )

    env.close()
    # Save test metrics to the metrics dictionary
    SARSA_metrics["test_success_count"] = success_count
    SARSA_metrics["test_failure_count"] = failure_count
    SARSA_metrics["successful_episodes"] = successful_episodes
    # Save updated metrics to metrics.json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training completed and metrics saved to metrics.json")

    # Print summary of successful and unsuccessful episodes
    print("\nSummary of Testing:")
    print(f"Total Successful Episodes: {success_count}")
    print(f"Total Unsuccessful Episodes: {failure_count}")
    if successful_episodes:
        print(
            f"Successful episodes recorded in files: {['video{}.mp4'.format(i) for i in successful_episodes]}"
        )
    else:
        print("No successful episodes recorded.")
