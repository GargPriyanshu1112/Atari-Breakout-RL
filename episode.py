import time
import numpy as np

"""Loop through each step of episode until it reach done flagl..."""
TARGET_UPDATE_PERIOD = 10000  # # make it  class parameter....


def get_new_state(state, obs):
    # print(np.append(state[:, :, 1:], np.expand_dims(obs, axis=-1), axis=-1).shape)
    return np.append(state[:, :, 1:], np.expand_dims(obs, axis=-1), axis=-1)


def learn(base_network, target_network, replay_memory, gamma):  # check dims
    states, actions, rewards, next_states, done_flags = replay_memory.get_batch()
    # print(
    #     states.shape, actions.shape, rewards.shape, next_states.shape, done_flags.shape
    # )

    # Get the target
    pred_Qs = target_network.predict(next_states)  # why target model??
    # print(f"pred_Qs shape: {pred_Qs.shape}")
    maxQs = np.max(pred_Qs, axis=1)
    # print(f"maxQs shape: {maxQs.shape}")
    targets = rewards + np.invert(done_flags).astype(np.float32) * gamma * maxQs
    # print(f"targets shape: {targets.shape}")

    loss = base_network.update(states, actions, targets)
    return loss


def play_episode(
    env,
    base_network,
    target_network,
    img_transformer,
    replay_memory,
    step_count,
    num_stacked_frames=4,
    gamma=0.99,
    eps=1.0,
    eps_min=0.1,
):
    start_time = time.time()

    obs, _ = env.reset()
    frame = img_transformer.transform(obs)
    state = np.stack([frame] * num_stacked_frames, axis=2)

    done = False
    episode_reward = 0
    num_episode_steps = 0
    eps_change = (eps - eps_min) / replay_memory.buffer_size
    while not done:
        # Update target network
        if step_count % TARGET_UPDATE_PERIOD == 0:
            print(f"Copying base model parameters to the target model...")
            target_network.copy_weights(base_network.model)

        action = base_network.sample_action(state, eps)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = img_transformer.transform(obs)
        new_state = get_new_state(state, frame)  # how random will help?
        assert new_state.shape == (
            img_transformer.h,
            img_transformer.w,
            num_stacked_frames,
        )
        replay_memory.add_experience(action, frame, reward, terminated or truncated)

        # Train
        loss = learn(base_network, target_network, replay_memory, gamma)

        end_time = time.time()
        duration = end_time - start_time

        state = new_state
        episode_reward += reward
        num_episode_steps += 1
        step_count += 1
        eps = max(eps_min, eps - eps_change)
        done = terminated or truncated

    return duration, loss, episode_reward, num_episode_steps, step_count
