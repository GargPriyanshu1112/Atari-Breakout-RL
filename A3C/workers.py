import threading
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from step import Step
from networks import ActorCritic
from utils import get_breakout_env
from image_transformer import ImageTransformer
from config import NUM_ACTIONS, NUM_STACKED_FRAMES


# Creates initial state by repeating the first frame 'NUM_STACKED_FRAMES' times
def repeat_frame(frame):
    return np.stack([frame] * NUM_STACKED_FRAMES, axis=-1)


# Returns next state by shifting each frame by 1. Removes the oldest frame from
# the state and concatenates the latest frame to its other end.
def get_next_state(state, frame):
    return np.append(state[:, :, 1:], np.expand_dims(frame, axis=-1), axis=-1)


class Worker:
    def __init__(
        self,
        name,
        global_network,
        actor_opt,
        critic_opt,
        global_step_counter,
        rewards_list,
        discount_factor=0.99,
        max_steps=5e6,
    ):
        self.name = name
        self.global_network = global_network
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.global_step_counter = global_step_counter
        self.rewards_list = rewards_list
        self.discount_factor = discount_factor
        self.max_steps = max_steps

        self.env = get_breakout_env()
        self.worker_network = ActorCritic()
        self.img_transformer = ImageTransformer()

        self.state = None  # tracks current state
        self.episode_reward = 0  # tracks episode reward

        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()

    def copy_global_weights(self):
        """Copies global Actor & Critic weights to the worker counterparts."""
        self.worker_network.actor.set_weights(self.global_network.actor.get_weights())
        self.worker_network.critic.set_weights(self.global_network.critic.get_weights())

    def update_global_weights(self, states, actions, advantages, value_targets):
        """Updates global Actor and Critic weights with gradients from worker counterparts."""
        with tf.GradientTape() as t:
            action_probs = self.worker_network.get_action_probs(states)  # p(a| s)
            chosen_action_probs = tf.reduce_sum(
                action_probs * tf.one_hot(actions, depth=NUM_ACTIONS), axis=1
            )
            loss = self.worker_network.actor_loss_fn(
                action_probs, chosen_action_probs, advantages
            )
        grads = t.gradient(loss, self.worker_network.actor.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)  # gradient clipping
        self.actor_opt.apply_gradients(
            zip(grads, self.global_network.actor.trainable_weights)
        )

        with tf.GradientTape() as t:
            preds = self.worker_network.get_v_estimate(states)
            loss = self.worker_network.critic_loss_fn(value_targets, preds)
        grads = t.gradient(loss, self.worker_network.critic.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)  # gradient clipping
        self.critic_opt.apply_gradients(
            zip(grads, self.global_network.critic.trainable_weights)
        )

    def sample_action(self, state):
        """Decides which action a worker will choose."""
        logits = self.worker_network.actor(np.array(state))
        distibution = tfp.distributions.Categorical(logits=logits)
        action = distibution.sample()[0]
        return action

    def run_n_steps(self, n):
        steps_data = []

        for _ in range(n):
            action = self.sample_action([self.state])  # Always starts at initial state

            obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = get_next_state(self.state, self.img_transformer.transform(obs))

            if terminated or truncated:  # if episode ends
                print(f"Episode Reward: {self.episode_reward} - {self.name}")
                self.rewards_list.append(self.episode_reward)
                self.episode_reward = 0  # reset
                if len(self.rewards_list) > 0 and len(self.rewards_list) % 100 == 0:
                    print(
                        f"\nAvg. reward (last 100 episodes): {np.mean(self.rewards_list[-100:])}\n"
                    )
            else:  # if episode hasn't ended
                self.episode_reward += reward

            # Update step counter
            num_global_steps = next(self.global_step_counter)

            # Store step data
            step = Step(self.state, action, reward, next_state, terminated or truncated)
            steps_data.append(step)

            if terminated or truncated:  # if episode ends, reset environment
                self.state = repeat_frame(
                    self.img_transformer.transform(self.env.reset()[0])
                )
                break
            else:  # if episode hasn't ended, go to next state
                self.state = next_state
        return steps_data, num_global_steps

    # TODO: stop threads, return to main function if `KeyboardInterrupt` occurs
    def run(self, coordinator, steps_before_update):
        # Start state
        self.state = repeat_frame(self.img_transformer.transform(self.env.reset()[0]))

        try:
            while not coordinator.should_stop():
                # Acquire the lock before entering the critical section
                self.lock1.acquire()
                try:
                    self.copy_global_weights()
                finally:
                    # Release the lock
                    self.lock1.release()

                # Collect experience
                steps_data, num_global_steps = self.run_n_steps(steps_before_update)

                if num_global_steps >= self.max_steps:
                    coordinator.request_stop()
                    return

                # Acquire the lock before entering the critical section
                self.lock2.acquire()
                try:
                    self.learn(steps_data)
                finally:
                    # Release the lock
                    self.lock2.release()

        except tf.errors.CancelledError:
            return

    def learn(self, steps_data):
        # If episode hasn't ended, get expected sum of all future rewards
        if not steps_data[-1].done_flag:
            s_prime = np.expand_dims(steps_data[-1].next_state, axis=0)
            V_s_prime = self.worker_network.get_v_estimate(s_prime)
        else:
            # If episode has ended, there will be no future rewards
            V_s_prime = 0.0

        states = []
        actions = []
        advantages = []
        value_targets = []

        """
        If we have s1, s2, s3 with rewards r1, r2, r3
        Then,  G(s3) = r3 + V(s4)
               G(s2) = r2 + r3 + V(s4)        (= r2 + G(s3))
               G(s1) = r1 + r2 + r3 + V(s4)   (= r1 + G(s2))
        """
        # Accumulated data for training batch
        for step_data in reversed(steps_data):
            # Get return
            G = step_data.reward + self.discount_factor * V_s_prime
            # Get V(s)
            s = np.expand_dims(step_data.state, axis=0)
            V_s = self.worker_network.get_v_estimate(s)
            # Get advantage
            advantage = G - V_s

            V_s_prime = G

            states.append(step_data.state)
            actions.append(step_data.action)
            advantages.append(advantage)
            value_targets.append(G)

        self.update_global_weights(
            np.array(states), actions, np.array(advantages), value_targets
        )
