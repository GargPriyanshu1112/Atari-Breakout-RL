# Import dependencies
import gym
import tensorflow as tf
from image_transformer import ImageTransformer
from dqn import DQN

state_history = []
action_history = []
reward_history = []
next_state_history = []
done_history = []

episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0



# while True:
#     start_state = env.reset()[0]
#     episode_reward = 0
#     for timestep in range(1, max_steps_per_episode):
#         frame_count += 1

#         if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
#             # Take random action
#             action = np.random.choice(num_actions)
#         else:
#             state_tensor = tf.convert_to_tensor(state)
#             state_tensor
          











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

    imgT = ImageTransformer()
    m = DQN()

# Prealocate all the frames we plan on storing and then we can sample states
# from the indivisual frames later on.
    

    BUFFER_SIZE = 5_00_000
    IMG_SIZE = 84

    class ReplayBuffer:
        def __init__(self, size=BUFFER_SIZE, frame_h=IMG_SIZE, framw_w=IMG_SIZE, agent_history_length=4, batch_size=32):

