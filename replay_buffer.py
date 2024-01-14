import numpy as np
import random

BUFFER_SIZE = 5_00_000
IMG_SIZE = 84


class ReplayBuffer:
    def __init__(
        self,
        size=BUFFER_SIZE,
        frame_h=IMG_SIZE,
        frame_w=IMG_SIZE,
        num_stacked_frames=4,
        batch_size=32,
    ):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.num_stacked_frames = num_stacked_frames
        self.batch_size = batch_size
        # To keep track of the insertion point in replay buffer
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(size, dtype=np.int32)
        self.rewards = np.empty(size, dtype=np.float32)
        self.frames = np.empty((size, frame_h, frame_w), dtype=np.float32)
        self.terminal_flags = np.empty(size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty(
            (batch_size, num_stacked_frames, frame_h, frame_w), dtype=np.float32
        )
        self.new_states = np.empty(
            (batch_size, num_stacked_frames, frame_h, frame_w), dtype=np.float32
        )
        self.indices = np.empty(batch_size, dtype=np.int32)

        def add_experience(self, action, frame, reward, terminal):
            assert frame.shape == (self.frame_h, self.frame_w)
            self.actions[self.current] = action
            self.frames[self.current] = frame
            self.rewards[self.current] = reward
            self.terminal_flags[self.current] = terminal
            self.current = (self.current + 1) % self.size
            self.count = max(self.count, self.current + 1)

        def get_state(self, idx):
            # idx represents final frame in the state. and return all the consecutive frames.

            if self.count is 0:
                raise ValueError
            if idx < self.num_stacked_frames - 1:
                raise ValueError
            return self.frames[idx - self.num_stacked_frames + 1 : idx + 1, ...]

        def get_valid_indices(self):
            """helper function to help a batch"""
            for i in range(self.batch_size):
                while True:
                    idx = random.randint(self.num_stacked_frames, self.count - 1)
                    if idx < self.num_stacked_frames:
                        continue
            pass

        ## Incomplete...
