# Import dependencies
import gym
import itertools

if __name__ == "__main__":
    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    env.reset()

    total_steps_counter = itertools.count()
