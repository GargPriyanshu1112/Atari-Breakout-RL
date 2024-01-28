import numpy as np


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
