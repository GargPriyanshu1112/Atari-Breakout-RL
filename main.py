# Import dependencies
import gym
import numpy as np
import tensorflow as tf
from image_transformer import ImageTransformer
from dqn import DQN
from replay_memory import ReplayMemory
from episode import play_episode

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

if __name__ == "__main__":
    GAMMA = 0.99
    BATCH_SIZE = 32
    NUM_EPISODES = 3500
    IMG_H = 84
    IMG_W = 84
    NUM_STACKED_FRAMES = 4  # no. of frames stacked together to make up one state
    REPLAY_BUFFER_SIZE = 500000
    MIN_BUFFER_SIZE = 50000  # minimum buffer size before commencing training
    MIN_STEPS_BEFORE_TARGET_UPDATE = (
        10000  # minimum steps before we update the target model's weights
    )
    EPS = 1.0
    EPS_MIN = 0.1

    base_model = DQN()
    target_model = DQN()
    img_transformer = ImageTransformer(IMG_H, IMG_W)
    replay_buffer = ReplayMemory(
        REPLAY_BUFFER_SIZE, IMG_H, IMG_W, BATCH_SIZE, NUM_STACKED_FRAMES
    )

    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    env.reset()

    # Populate replay buffer with episodes of completely random actions
    for _ in range(
        MIN_BUFFER_SIZE
    ):  # ?? min buffer size or REPLAT_BUFFER SIZE?? ---- why random ????
        action = np.random.choice(env.action_space.n)
        frame, reward, terminated, truncated, _ = env.step(action)
        processed_frame = img_transformer.transform(frame)
        replay_buffer.add_experience(
            action, processed_frame, reward, terminated or truncated
        )

        if terminated or truncated:
            env.reset()

    total_steps = 0
    rewards_per_episode = []
    # Play episodes and learn...
    for i in range(NUM_EPISODES):  # ??
        duration, loss, episode_reward, num_episode_steps, total_steps = play_episode(
            env,
            img_transformer,
            base_model,
            target_model,
            replay_buffer,
            total_steps,
            BATCH_SIZE,
            NUM_STACKED_FRAMES,
            GAMMA,
            EPS,
            EPS_MIN,
        )

        print(
            f"episode {i+1} | duration: {duration} sec. | loss: {loss} | reward: {episode_reward} | steps: {num_episode_steps}"
        )

        rewards_per_episode.append(episode_reward)

    # Final touches...
