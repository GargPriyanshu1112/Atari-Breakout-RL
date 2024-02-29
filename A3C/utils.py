import gym
import numpy as np
import matplotlib.pyplot as plt


def get_breakout_env():
    # https://www.gymlibrary.dev/environments/atari/index.html
    return gym.make("Breakout-v0")


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


def plot_results(rewards):
    smoothed_rewards = smooth(rewards)

    plt.title("Episodes vs Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.savefig("results.png")
    plt.show()
