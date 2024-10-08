import json
import matplotlib.pyplot as plt
import os

# Load metrics from the metrics.json file
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Extract Q-learning and SARSA metrics
q_learning_metrics = metrics["Q_learning"]
sarsa_metrics = metrics["SARSA"]

# Create a directory to save the plots if it doesn't already exist
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# First successful episodes for Q-learning and SARSA
q_learning_first_success = q_learning_metrics.get("first_success_episode", None)
sarsa_first_success = sarsa_metrics.get("first_success_episode", None)

# 1.1. Plot Total Rewards per Episode (0 to 5000) for Q-learning and SARSA
plt.figure(figsize=(12, 6))
plt.plot(
    q_learning_metrics["total_rewards"][:5000],
    label="Q-learning - Total Rewards",
    color="b",
)
plt.plot(
    sarsa_metrics["total_rewards"][:5000], label="SARSA - Total Rewards", color="g"
)

# Add markers for the first success episodes
if q_learning_first_success is not None and q_learning_first_success < 5000:
    plt.axvline(
        q_learning_first_success,
        color="b",
        linestyle="--",
        label="Q-learning First Success",
        alpha=0.7,
    )
if sarsa_first_success is not None and sarsa_first_success < 5000:
    plt.axvline(
        sarsa_first_success,
        color="g",
        linestyle="--",
        label="SARSA First Success",
        alpha=0.7,
    )

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Rewards per Episode (0 to 5000): Q-learning vs SARSA")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "total_rewards_0_to_5000_comparison.png"))
plt.close()

# 1.2. Plot Total Rewards per Episode (5001 onwards) for Q-learning and SARSA
plt.figure(figsize=(12, 6))
plt.plot(
    range(5000, len(q_learning_metrics["total_rewards"])),
    q_learning_metrics["total_rewards"][5000:],
    label="Q-learning - Total Rewards",
    color="b",
)
plt.plot(
    range(5000, len(sarsa_metrics["total_rewards"])),
    sarsa_metrics["total_rewards"][5000:],
    label="SARSA - Total Rewards",
    color="g",
)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Rewards per Episode (5001 onwards): Q-learning vs SARSA")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "total_rewards_5001_onwards_comparison.png"))
plt.close()

# 2. Find Convergence Point for Number of Steps per Episode
CONVERGENCE_THRESHOLD_STEPS = 1000


def find_convergence_point(steps_per_episode, threshold):
    for i in range(len(steps_per_episode)):
        if steps_per_episode[i] <= threshold:
            return i
    return len(steps_per_episode)  # If no convergence point is found


# Find convergence points for Q-learning and SARSA (Number of Steps)
q_learning_convergence_point = find_convergence_point(
    q_learning_metrics["steps_per_episode"], CONVERGENCE_THRESHOLD_STEPS
)
sarsa_convergence_point = find_convergence_point(
    sarsa_metrics["steps_per_episode"], CONVERGENCE_THRESHOLD_STEPS
)

# 2.1. Plot Number of Steps per Episode (until convergence) for Q-learning and SARSA
plt.figure(figsize=(12, 6))
plt.plot(
    range(q_learning_convergence_point + 1),
    q_learning_metrics["steps_per_episode"][: q_learning_convergence_point + 1],
    label="Q-learning - Steps per Episode",
    color="b",
)
plt.plot(
    range(sarsa_convergence_point + 1),
    sarsa_metrics["steps_per_episode"][: sarsa_convergence_point + 1],
    label="SARSA - Steps per Episode",
    color="g",
)

# Add markers for the first success episodes
if (
    q_learning_first_success is not None
    and q_learning_first_success <= q_learning_convergence_point
):
    plt.axvline(
        q_learning_first_success,
        color="b",
        linestyle="--",
        label="Q-learning First Success",
        alpha=0.7,
    )
if sarsa_first_success is not None and sarsa_first_success <= sarsa_convergence_point:
    plt.axvline(
        sarsa_first_success,
        color="g",
        linestyle="--",
        label="SARSA First Success",
        alpha=0.7,
    )

plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.title("Number of Steps per Episode (Until Convergence): Q-learning vs SARSA")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "steps_until_convergence_comparison.png"))
plt.close()

# Print information about the convergence points
print(f"Q-learning first converged around episode: {q_learning_convergence_point}")
print(f"SARSA first converged around episode: {sarsa_convergence_point}")

# 2.2. Plot Full Number of Steps per Episode for Q-learning and SARSA
plt.figure(figsize=(12, 6))
plt.plot(
    q_learning_metrics["steps_per_episode"],
    label="Q-learning - Steps per Episode",
    color="b",
)
plt.plot(
    sarsa_metrics["steps_per_episode"], label="SARSA - Steps per Episode", color="g"
)

# Add markers for the first success episodes
if q_learning_first_success is not None:
    plt.axvline(
        q_learning_first_success,
        color="b",
        linestyle="--",
        label="Q-learning First Success",
        alpha=0.7,
    )
if sarsa_first_success is not None:
    plt.axvline(
        sarsa_first_success,
        color="g",
        linestyle="--",
        label="SARSA First Success",
        alpha=0.7,
    )

plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.title("Number of Steps per Episode (Full Data): Q-learning vs SARSA")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "steps_full_data_comparison.png"))
plt.close()

# 3. Plot Epsilon Decay Over Time for Q-learning and SARSA
plt.figure(figsize=(12, 6))
plt.plot(
    q_learning_metrics["epsilon_values"], label="Q-learning - Epsilon Value", color="b"
)
plt.plot(sarsa_metrics["epsilon_values"], label="SARSA - Epsilon Value", color="g")
plt.xlabel("Episode")
plt.ylabel("Epsilon Value")
plt.title("Epsilon Decay Over Time: Q-learning vs SARSA")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "epsilon_decay_comparison.png"))
plt.close()

# 4. Success Counts Summary (Bar Plot)
labels = ["Q-learning", "SARSA"]
success_counts = [q_learning_metrics["success_count"], sarsa_metrics["success_count"]]
test_success_counts = [
    q_learning_metrics["test_success_count"],
    sarsa_metrics["test_success_count"],
]

x = range(len(labels))

plt.figure(figsize=(10, 6))
plt.bar(
    x,
    success_counts,
    width=0.4,
    label="Training Success Count",
    color="blue",
    align="center",
)
plt.bar(
    x,
    test_success_counts,
    width=0.4,
    label="Testing Success Count",
    color="green",
    align="edge",
)
plt.xlabel("Algorithm")
plt.ylabel("Number of Successful Episodes")
plt.title("Comparison of Success Counts: Q-learning vs SARSA")
plt.xticks(x, labels)
plt.legend()
plt.grid(axis="y")
plt.savefig(os.path.join(plot_dir, "success_counts_comparison.png"))
plt.close()

# 5. Find Convergence Point for Training Time per Episode (until close to 0.1 sec)
CONVERGENCE_THRESHOLD_TIME = 0.1


def find_time_convergence_point(training_time_per_episode, threshold):
    for i in range(len(training_time_per_episode)):
        if training_time_per_episode[i] <= threshold:
            return i
    return len(training_time_per_episode)  # If no convergence point is found


# Find convergence points for Q-learning and SARSA (Training Time)
q_learning_time_convergence_point = find_time_convergence_point(
    q_learning_metrics["training_time_per_episode"], CONVERGENCE_THRESHOLD_TIME
)
sarsa_time_convergence_point = find_time_convergence_point(
    sarsa_metrics["training_time_per_episode"], CONVERGENCE_THRESHOLD_TIME
)
# 5.1. Plot Training Time per Episode (until convergence) for Q-learning and SARSA
plt.figure(figsize=(12, 6))
plt.plot(
    range(q_learning_time_convergence_point + 1),
    q_learning_metrics["training_time_per_episode"][
        : q_learning_time_convergence_point + 1
    ],
    label="Q-learning - Training Time per Episode",
    color="b",
)
plt.plot(
    range(sarsa_time_convergence_point + 1),
    sarsa_metrics["training_time_per_episode"][: sarsa_time_convergence_point + 1],
    label="SARSA - Training Time per Episode",
    color="g",
)
plt.xlabel("Episode")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time per Episode (Until Convergence): Q-learning vs SARSA")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "training_time_until_convergence_comparison.png"))
plt.close()

# Print information about the convergence points for training time
print(
    f"Q-learning first converged in training time around episode: {q_learning_time_convergence_point}"
)
print(
    f"SARSA first converged in training time around episode: {sarsa_time_convergence_point}"
)

print(f"All plots have been saved to the '{plot_dir}' directory.")
