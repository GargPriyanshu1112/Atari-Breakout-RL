# Import dependencies
import gym
import tensorflow as tf
from image_transformer import ImageTransformer


if __name__ == "__main__":
    print(tf.__version__)
    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    start_state, info = env.reset()

    # print(start_state.shape)
    # print(info)

    imgT = ImageTransformer()
    m = DQN()
    m.model.summary()
