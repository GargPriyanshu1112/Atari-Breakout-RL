import numpy as np

from config import Config


def repeat_frame(frame):
    return np.stack([frame] * Config.NUM_STACKED_FRAMES, axis=-1)


def get_next_state(state, frame):
    return np.append(state[:, :, 1:], np.expand_dims(frame, axis=-1), axis=-1)


def get_networks():
    """
    Returns Policy and Value model.
    """
    pass


def copy_params():
    """
    Copies params from global model to worker model (only the base layers ??)
    """


def update_weights():
    """
    Updates
    """


class Step:
    def __init__(self, current_state, action, reward, next_state, done_flag):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done_flag = done_flag
