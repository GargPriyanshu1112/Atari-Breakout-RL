import numpy as np
import matplotlib.pyplot as plt


## -------------------------------- Plotting Functions -------------------------------- ##


def smooth(x):
    """
    Returns average over last 100 episodes
    """
    x = np.array(x)  # list -> numpy array
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start : (i + 1)].sum()) / (i - start + 1)
    return y


def plot_results(episode_rewards, episode_steps):
    smoothed_rewards = smooth(episode_rewards)
    smoothed_step_count = smooth(episode_steps)

    plt.subplot(1, 2, 1)
    plt.title("Episodes vs Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.plot(episode_rewards)
    plt.plot(smoothed_rewards)

    plt.subplot(1, 2, 2)
    plt.title("Episode vs Step Count")
    plt.xlabel("Episodes")
    plt.ylabel("Step Count")
    plt.plot(smoothed_step_count)
    plt.savefig("results.png")
    plt.show()


## ------------------------------------------------------------------------------------ ##
