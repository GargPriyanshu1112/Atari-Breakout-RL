import numpy as np
import random


# Pre-allocate all of the frames we plan on storing and then we can sample
# states from the individual frames later on.


class ReplayMemory:
    def __init__(
        self,
        buffer_size,
        frame_h,
        frame_w,
        batch_size,
        num_stacked_frames=4,  # no. of frames stacked together to make up one state
    ):
        self.batch_size = batch_size
        self.num_stacked_frames = num_stacked_frames
        self.current_buffer_pos = 0
        self.count = 0

        # Pre-allocate memory for replay buffer
        self.actions = np.empty(buffer_size, dtype=np.int32)
        self.rewards = np.empty(buffer_size, dtype=np.float32)
        self.frames = np.empty((buffer_size, frame_h, frame_w), dtype=np.float32)
        self.terminal_flags = np.empty(buffer_size, dtype=np.bool)

        # Pre-allocate memory for a minibatch
        self.states = np.empty(
            (batch_size, num_stacked_frames, frame_h, frame_w), dtype=np.float32
        )
        self.new_states = np.empty(
            (batch_size, num_stacked_frames, frame_h, frame_w), dtype=np.float32
        )
        self.indices = np.empty(batch_size, dtype=np.int32)

        def add_experience(self, action, frame, reward, is_terminal):  # is_terminal ??
            assert frame.shape == (self.frame_h, self.frame_w)
            self.actions[self.current] = action
            self.frames[self.current] = frame
            self.rewards[self.current] = reward
            self.terminal_flags[self.current] = is_terminal
            self.count = max(self.count, self.current + 1)  # use ??
            self.current = (self.current + 1) % self.size  # circular insertion

        def get_state(self, frame_idx):
            if frame_idx < self.num_stacked_frames - 1:
                raise ValueError("`frame_idx` cannot be less than 3.")
            if self.count == 0:
                raise ValueError("The replay memory is empty.")
            return self.frames[
                frame_idx + 1 - self.num_stacked_frames : frame_idx + 1, ...
            ]

        def get_valid_indices(self):
            """helper function to sample a batch"""
            for i in range(self.batch_size):
                while True:
                    idx = random.randint(
                        self.num_stacked_frames, self.count
                    )  # -1 removed ??

                    if self.terminal_flags[idx - self.num_stacked_frames : idx].any():
                        continue
                    elif (
                        idx >= self.current_buffer_pos
                        and idx - self.num_stacked_frames <= self.current_buffer_pos
                    ):
                        continue
                    break
                self.indices[i] = idx

        def get_batch(self):
            """Returns a batch of self.batch_size transitions."""
            self.get_valid_indices()

            for i, idx in enumerate(self.indices):
                self.states[i] = self.get_state(idx - 1)  # ??
                self.new_states[i] = self.get_state(idx)  # ??

            return ()