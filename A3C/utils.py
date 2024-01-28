import numpy as np

from config import Config


def repeat_frame(frame):
    return np.stack([frame] * Config.NUM_STACKED_FRAMES, axis=-1)


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
