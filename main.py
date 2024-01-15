# Import dependencies
import gym
import numpy as np
import tensorflow as tf
from image_transformer import ImageTransformer
from dqn import DQN
from replay_memory import ReplayMemory

state_history = []
action_history = []
reward_history = []
next_state_history = []
done_history = []

episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0


TARGET_UPDATE_PERIOD = None


def get_next_state(state, obs):
    return np.append(state[:, :, 1:], np.expand_dims(obs, axis=-1), axis=-1)


def learn(base_model, target_model, replay_buffer, gamma, batch_size):
    replay_buffer...


def play_episode(
    env,
    img_transformer,
    num_stacked_frames,
    time_step,
    replay_buffer,
    base_model,
    target_model,
    gamma,
    batch_size,
    eps,
    epsilon_change,
    epsilon_min,
):
    obs, _ = env.reset()
    obs = img_transformer.transform(obs)
    state = np.stack([obs] * num_stacked_frames, axis=2)

    done = False
    while not done:
        # Update target network
        if time_step % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_weights(base_model)

        action = base_model.sample_action(state, eps)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = img_transformer.transform(obs)
        next_state = get_next_state(state, obs)
        is_terminal = terminated or truncated
        replay_buffer.add_experience(action, obs, reward, is_terminal)
        loss = learn(base_model, target_model, replay_buffer, gamma, batch_size)

        state = next_state


if __name__ == "__main__":
    GAMMA = 0.99
    BATCH_SIZE = 32 ??
    NUM_EPISODES = 3500
    IMG_SIZE = 84
    REPLAY_BUFFER_SIZE = 500000
    MIN_BUFFER_SIZE = 50000  # minimum buffer size before commencing training
    MIN_STEPS_BEFORE_TARGET_UPDATE = 10000 # minimum steps before we update the target model's weights 
    EPSILON = 1.0
    EPSILON_MIN = 0.1

    base_model = DQN()
    target_model = DQN()
    img_transformer = ImageTransformer(IMG_SIZE)
    replay_buffer = ReplayMemory(REPLAY_BUFFER_SIZE, IMG_SIZE, IMG_SIZE, BATCH_SIZE)

    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    env.reset()

    # Populate replay buffer with episodes of completely random actions
    for _ in range(MIN_BUFFER_SIZE):  # ?? min buffer size or??
        action = np.random.choice(env.action_space.n)
        frame, reward, terminated, truncated, _ = env.step(action)
        processed_frame = img_transformer.transform(frame)
        is_terminal = terminated or truncated
        replay_buffer.add_experience(action, processed_frame, reward, is_terminal)

        if is_terminal:
            env.reset()

    # Play episodes and learn...
    for _ in range(NUM_EPISODES):  # ??
        play_episode()

    # Prealocate all the frames we plan on storing and then we can sample states
    # from the indivisual frames later on.

    BUFFER_SIZE = 5_00_000
    IMG_SIZE = 84

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
