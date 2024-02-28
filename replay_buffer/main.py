# Import dependencies
import gym
import numpy as np
from image_transformer import ImageTransformer
from dqn import DQN
from replay_memory import ReplayMemory
from episode import play_episode
from utils import smooth, plot_results


if __name__ == "__main__":
    # https://www.gymlibrary.dev/environments/atari/index.html
    # Initialize the Breakout environment
    env = gym.make("Breakout-v0")
    env.reset()

    NUM_ACTIONS = env.action_space.n
    assert NUM_ACTIONS == 4

    H = 84  # frame height
    W = 84  # frame width
    NUM_STACKED_FRAMES = 4  # no. of frames stacked together to make up one state
    INP_SHAPE = (H, W, NUM_STACKED_FRAMES)

    REPLAY_BUFFER_SIZE = 500000  # max buffer size
    MIN_BUFFER_SIZE = 50000  # minimum buffer size before commencing training
    TARGET_UPDATE_PERIOD = 10000  # steps after which target model's weights are updated

    EPS = 1.0
    EPS_MIN = 0.1
    EPS_CHANGE = (EPS - EPS_MIN) / REPLAY_BUFFER_SIZE
    GAMMA = 0.99
    BATCH_SIZE = 32
    NUM_EPISODES = 3500

    main_network = DQN(INP_SHAPE, NUM_ACTIONS)
    target_network = DQN(INP_SHAPE, NUM_ACTIONS)
    img_transformer = ImageTransformer(H, W)
    replay_memory = ReplayMemory(
        REPLAY_BUFFER_SIZE, H, W, NUM_STACKED_FRAMES, BATCH_SIZE
    )

    # Populate replay buffer with episodes of completely random actions
    for _ in range(MIN_BUFFER_SIZE):
        action = np.random.choice(NUM_ACTIONS)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = img_transformer.transform(obs)
        replay_memory.add_experience(action, frame, reward, terminated or truncated)

        if terminated or truncated:
            env.reset()

    step_count = 0
    steps_per_episode = []
    rewards_per_episode = []
    # Play episodes
    for i in range(NUM_EPISODES):
        duration, episode_reward, num_episode_steps, step_count, EPS = play_episode(
            env,
            main_network,
            target_network,
            img_transformer,
            replay_memory,
            step_count,
            EPS,
            EPS_MIN,
            EPS_CHANGE,
            NUM_STACKED_FRAMES,
            GAMMA,
            TARGET_UPDATE_PERIOD,
        )

        rewards_per_episode.append(episode_reward)
        steps_per_episode.append(num_episode_steps)

        print(
            f"Episode {i+1} | "
            f"Steps: {num_episode_steps} | "
            f"Reward: {episode_reward} | "
            f"Epsilon: {EPS:.4f} | "
            f"Duration: {duration:.2f} sec.| "
            f"Avg. reward (last 100 episodes): {smooth(rewards_per_episode)[-1]:.2f} | "
            f"Total steps: {step_count}"
        )

    # Save model for inference
    main_network.model.save("model.keras")
    # Plot results
    plot_results(rewards_per_episode, steps_per_episode)
