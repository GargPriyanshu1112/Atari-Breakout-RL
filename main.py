# Import dependencies
import gym
import numpy as np
import tensorflow as tf
from image_transformer import ImageTransformer
from dqn import DQN
from replay_memory import ReplayMemory
from episode import play_episode


if __name__ == "__main__":
    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    env.reset()

    NUM_ACTIONS = env.action_space.n
    GAMMA = 0.99
    BATCH_SIZE = 32
    NUM_EPISODES = 3500
    IMG_H = 84
    IMG_W = 84
    NUM_STACKED_FRAMES = 4  # no. of frames stacked together to make up one state
    REPLAY_BUFFER_SIZE = 500000
    MIN_BUFFER_SIZE = 50000  # minimum buffer size before commencing training 00
    MIN_STEPS_BEFORE_TARGET_UPDATE = (
        10000  # minimum steps before we update the target model's weights
    )
    EPS = 1.0
    EPS_MIN = 0.1

    INP_SHAPE = (IMG_H, IMG_W, NUM_STACKED_FRAMES)

    base_network = DQN(NUM_ACTIONS, INP_SHAPE)
    target_network = DQN(NUM_ACTIONS, INP_SHAPE)
    img_transformer = ImageTransformer(IMG_H, IMG_W)
    replay_memory = ReplayMemory(
        REPLAY_BUFFER_SIZE, IMG_H, IMG_W, BATCH_SIZE, NUM_STACKED_FRAMES
    )

    # Populate replay buffer with episodes of completely random actions
    for _ in range(
        MIN_BUFFER_SIZE
    ):  # ?? min buffer size or REPLAT_BUFFER SIZE?? ---- why random ????
        action = np.random.choice(env.action_space.n)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = img_transformer.transform(obs)
        replay_memory.add_experience(action, frame, reward, terminated or truncated)

        if terminated or truncated:
            env.reset()

    step_count = 0
    rewards_per_episode = []
    # Play episodes
    for i in range(NUM_EPISODES):
        duration, loss, episode_reward, num_episode_steps, step_count = play_episode(
            env,
            base_network,
            target_network,
            img_transformer,
            replay_memory,
            step_count,
            NUM_STACKED_FRAMES,
            GAMMA,
            EPS,
            EPS_MIN,
        )

        print(
            f"episode {i+1} | episode duration: {duration} sec. | loss: {loss} | reward: {episode_reward} | steps: {step_count}"
        )

        rewards_per_episode.append(episode_reward)

    # Final touches...
