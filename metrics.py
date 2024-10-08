import json
import os

# Define the structure of metrics.json with separate sections for Q-learning and SARSA
metrics = {
    "Q_learning": {
        "total_rewards": [],  # List to store total rewards per episode during Q-learning training
        "steps_per_episode": [],  # List to store steps taken per episode during Q-learning training
        "success_count": 0,  # Count of successful episodes in Q-learning training
        "epsilon_values": [],  # List to track epsilon values over episodes in Q-learning training
        "training_time_per_episode": [],  # List to store training time per episode during Q-learning training
        "test_success_count": 0,  # Count of successful episodes during Q-learning testing
        "test_failure_count": 0,  # Count of failed episodes during Q-learning testing
        "successful_episodes": [],  # List of successful episode indices during Q-learning testing
        "first_success_episode": None,  # Episode number where Q-learning first succeeded
    },
    "SARSA": {
        "total_rewards": [],  # List to store total rewards per episode during SARSA training
        "steps_per_episode": [],  # List to store steps taken per episode during SARSA training
        "success_count": 0,  # Count of successful episodes in SARSA training
        "epsilon_values": [],  # List to track epsilon values over episodes in SARSA training
        "training_time_per_episode": [],  # List to store training time per episode during SARSA training
        "test_success_count": 0,  # Count of successful episodes during SARSA testing
        "test_failure_count": 0,  # Count of failed episodes during SARSA testing
        "successful_episodes": [],  # List of successful episode indices during SARSA testing
        "first_success_episode": None,  # Episode number where SARSA first succeeded
    },
}

# File path to save metrics.json
file_path = "metrics.json"

# Create or overwrite the metrics.json file
with open(file_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(
    f"{file_path} created successfully with the initial structure for Q-learning and SARSA."
)

# Verify that the file is created and its structure
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))
