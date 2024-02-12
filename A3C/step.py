class Step:
    """Stores data related to each step in the environment."""

    def __init__(self, state, action, reward, next_state, done_flag):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done_flag = done_flag
