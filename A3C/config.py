import gym
import multiprocessing

env = gym.make(
    id="ALE/Breakout-v5",
    full_action_space=False,
    repeat_action_probability=0.1,
    obs_type="rgb",
)


MAX_STEPS = 5e6
MIN_STEPS_BEFORE_UPDATE = 5  # no. of steps each worker has to perform before calculating the gradient and sending it back to the global network
NUM_ACTIONS = 4
NUM_WORKERS = multiprocessing.cpu_count()
DISCOUNT_FACTOR = 0.99
NUM_STACKED_FRAMES = 4


class config:
    MAX_STEPS = 5e6
    MIN_STEPS_BEFORE_UPDATE = 5  # no. of steps each worker has to perform before calculating the gradient and sending it back to the global network
    NUM_ACTIONS = 4
    NUM_WORKERS = multiprocessing.cpu_count()
    DISCOUNT_FACTOR = 0.99
    NUM_STACKED_FRAMES = 4
