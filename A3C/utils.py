import gym
import numpy as np

from config import NUM_STACKED_FRAMES


# Returns Breakout environment
def get_breakout_env():
    return gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )


# Creates initial state by repeating the first frame 'NUM_STACKED_FRAMES' times
def repeat_frame(frame):
    return np.stack([frame] * NUM_STACKED_FRAMES, axis=-1)


# Returns next state by shifting each frame by 1. Removes the oldest frame from
# the state and concatenates the latest frame to its other end.
def get_next_state(state, frame):
    return np.append(state[:, :, 1:], np.expand_dims(frame, axis=-1), axis=-1)


# Class that stores data related to each step in the environment
class Step:
    def __init__(self, state, action, reward, next_state, done_flag):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done_flag = done_flag
