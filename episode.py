from time import time
import numpy as np

"""Loop through each step of episode until it reach done flagl..."""
TARGET_UPDATE_PERIOD = 10000  # # make it  class parameter....


def update_state(state, obs):
    return np.append(state[:, :, 1:], np.expand_dims(obs, axis=-1), axis=-1)


def learn(base_model, target_model, replay_buffer, gamma, batch_size):  # check dims
    states, actions, rewards, next_states, done_flags = replay_buffer.get_batch()

    # Get the target
    pred_Qs = target_model.predict(next_states)  # why target model??
    maxQs = np.max(pred_Qs, axis=1)  # check dimenstion...
    targets = rewards + np.invert(done_flags).astype(np.float) * gamma * maxQs

    loss = base_model.update(states, actions, targets)
    return loss


def play_episode(
    env,
    img_transformer,
    base_model,
    target_model,
    replay_buffer,
    num_total_steps,
    batch_size=32,
    num_stacked_frames=4,
    gamma=0.99,
    eps=1.0,
    eps_min=0.1,
):
    start_time = time.start()

    obs, _ = env.reset()
    obs = img_transformer.transform(obs)
    state = np.stack([obs] * num_stacked_frames, axis=2)  # first state

    num_episode_steps = episode_reward = 0
    done = False
    eps_change = (eps - eps_min) / replay_buffer.buffer_size
    while not done:
        # Update target network
        if num_total_steps % TARGET_UPDATE_PERIOD == 0:
            print(f"Copying base model parameters to the target model...")
            target_model.copy_weights(base_model)

        action = base_model.sample_action(state, eps)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = img_transformer.transform(obs)
        next_state = update_state(state, obs)  # how random will help?
        assert next_state.shape == (
            img_transformer.h,
            img_transformer.w,
            num_stacked_frames,
        )
        replay_buffer.add_experience(action, obs, reward, terminated or truncated)

        # Train
        loss = learn(base_model, target_model, replay_buffer, gamma, batch_size)

        state = next_state
        episode_reward += reward

        num_episode_steps += 1
        num_total_steps += 1

        end_time = time.end()
        duration = end_time - start_time

        eps = max(eps_min, eps - eps_change)

        return duration, loss, episode_reward, num_episode_steps, num_total_steps
