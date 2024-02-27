import time
import numpy as np


# Returns next state by shifting each frame by 1. Removes the oldest frame from
# the state and concatenates the latest frame to its other end.
def get_next_state(state, frame):
    new_state = np.append(state[:, :, 1:], np.expand_dims(frame, axis=-1), axis=-1)
    assert new_state.ndim == 3  # new_state.shape == (H, W, NUM_STACKED_FRAMES)
    return new_state


def learn(main_network, target_network, replay_memory, gamma):
    # Get train batch
    states, actions, rewards, next_states, done_flags = replay_memory.get_batch()
    # print(
    #     states.shape, actions.shape, rewards.shape, next_states.shape, done_flags.shape
    # )

    # Get the target
    next_Qvals = target_network.predict(next_states)
    # print(f"pred_Qs shape: {pred_Qs.shape}")
    nextQs = np.max(next_Qvals, axis=1)
    # print(f"maxQs shape: {maxQs.shape}")
    targets = rewards + np.invert(done_flags).astype(np.float32) * gamma * nextQs
    # print(f"targets shape: {targets.shape}")
    loss = main_network.update(states, actions, targets)
    return loss


def play_episode(
    env,
    main_network,
    target_network,
    img_transformer,
    replay_memory,
    step_count,
    eps,
    eps_min,
    eps_change,
    num_stacked_frames=4,
    gamma=0.99,
    target_update_period=10000,
):
    start_time = time.time()

    # Get initial episode state
    obs, _ = env.reset()
    frame = img_transformer.transform(obs)
    state = np.stack([frame] * num_stacked_frames, axis=-1)
    assert state.ndim == 3

    done = False
    episode_reward = 0
    num_episode_steps = 0
    while not done:
        # Update target network
        if step_count % target_update_period == 0:
            print(f"\nCopying base model parameters to the target model...\n")
            target_network.copy_weights(main_network.model)

        action = main_network.sample_action(state, eps)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = img_transformer.transform(obs)
        next_state = get_next_state(state, frame)  # how random will help?
        replay_memory.add_experience(action, frame, reward, terminated or truncated)

        # Train
        loss = learn(main_network, target_network, replay_memory, gamma)

        state = next_state
        eps = max(eps_min, eps - eps_change)
        done = terminated or truncated
        episode_reward += reward
        num_episode_steps += 1
        step_count += 1

    end_time = time.time()
    duration = end_time - start_time

    return duration, episode_reward, num_episode_steps, step_count, eps
