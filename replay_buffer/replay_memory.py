import numpy as np
import random


class ReplayMemory:
    def __init__(
        self,
        buffer_size=500000,
        frame_h=84,
        frame_w=84,
        batch_size=32,
        num_stacked_frames=4,
    ):
        self.buffer_size = buffer_size
        self.h = frame_h
        self.w = frame_w
        self.batch_size = batch_size
        self.num_stacked_frames = num_stacked_frames
        self.buffer_pos = 0
        self.count = 0

        # Pre-allocate memory for replay buffer
        self.actions = np.empty(buffer_size, dtype=np.int32)
        self.rewards = np.empty(buffer_size, dtype=np.float32)
        self.frames = np.empty((buffer_size, frame_h, frame_w), dtype=np.float32)
        self.done_flags = np.empty(buffer_size, dtype=np.bool_)

        # Pre-allocate memory for a minibatch
        self.states = np.empty(
            (batch_size, num_stacked_frames, frame_h, frame_w), dtype=np.float32
        )
        self.next_states = np.empty(
            (batch_size, num_stacked_frames, frame_h, frame_w), dtype=np.float32
        )
        self.indices = np.empty(batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, done_flag):
        self.actions[self.buffer_pos] = action
        self.frames[self.buffer_pos] = frame
        self.rewards[self.buffer_pos] = reward
        self.done_flags[self.buffer_pos] = done_flag

        # Update counter
        self.count = max(self.count, self.buffer_pos + 1)
        # Update current buffer position
        self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size

    def get_valid_indices(self):
        """Helper function to sample a batch"""
        for i in range(self.batch_size):
            while True:
                idx = random.randint(self.num_stacked_frames, self.count - 1)

                if self.done_flags[idx - self.num_stacked_frames : idx].any():
                    continue
                elif (
                    idx >= self.buffer_pos
                    and idx - self.num_stacked_frames <= self.buffer_pos
                ):
                    continue
                break
            self.indices[i] = idx

    def get_state(self, frame_idx):
        if frame_idx < self.num_stacked_frames - 1:
            raise ValueError("`frame_idx` cannot be less than 3.")
        if self.count == 0:
            raise ValueError("The replay memory is empty.")
        return self.frames[frame_idx + 1 - self.num_stacked_frames : frame_idx + 1, ...]

    def get_batch(self):
        """Returns a batch of state transitions."""
        self.get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self.get_state(idx - 1)  # ??
            self.next_states[i] = self.get_state(idx)  # ??

        return (
            np.transpose(self.states, axes=[0, 2, 3, 1]),
            self.actions[self.indices],
            self.rewards[self.indices],
            np.transpose(self.next_states, axes=[0, 2, 3, 1]),
            self.done_flags[self.indices],
        )
