import numpy as np
import tensorflow as tf

from image_transformer import ImageTransformer
from networks import get_networks
from utils import get_breakout_env, repeat_frame, get_next_state
from step import Step
from config import INP_SHAPE, NUM_ACTIONS


class Worker:
    def __init__(
        self,
        name,
        steps_counter,
        global_value_network,
        global_policy_network,
        returns_list,
        discount_factor=0.99,
        max_steps=5e6,
    ):
        self.name = name
        self.steps_counter = steps_counter
        self.global_value_network = global_value_network
        self.global_policy_network = global_policy_network
        self.returns_list = returns_list
        self.discount_factor = discount_factor
        self.max_steps = max_steps

        self.state = None  # tracks current state
        self.episode_reward = 0  # tracks total episode reward
        self.env = get_breakout_env()
        self.img_transformer = ImageTransformer()
        self.worker_value_network, self.worker_policy_network = get_networks(
            INP_SHAPE, NUM_ACTIONS
        )

    def copy_global_weights(self):
        """Copies weights from global value and policy models to the worker counterparts."""
        self.worker_value_network.model.set_weights(
            self.global_value_network.model.get_weights()
        )
        self.worker_policy_network.model.set_weights(
            self.global_policy_network.model.get_weights()
        )

    def update_global_weights(self, states, actions, advantages, value_targets):
        """Updates weights of global value and policy models with gradients from the worker counterparts."""
        with tf.GradientTape() as t:
            preds = self.worker_value_network.predict(states)
            loss = self.worker_value_network.loss_fn(value_targets, preds)
        gradients = t.gradient(loss, self.worker_value_network.model.trainable_weights)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # gradient clipping
        self.global_value_network.optimizer.apply_gradients(
            zip(gradients, self.global_value_network.model.trainable_weights)
        )

        with tf.GradientTape() as t:
            action_probs = self.worker_policy_network.get_probs(states)  # p(a| s)
            selected_action_probs = tf.reduce_sum(
                tf.multiply(action_probs, tf.one_hot(actions, depth=4)), axis=1
            )
            loss = self.worker_policy_network.loss_fn(
                action_probs, selected_action_probs, advantages
            )
        gradients = t.gradient(loss, self.worker_policy_network.model.trainable_weights)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # gradient clipping
        self.global_policy_network.optimizer.apply_gradients(
            zip(gradients, self.global_policy_network.model.trainable_weights)
        )

    def sample_action(self, state):
        return self.worker_policy_network.sample_action(np.array(state))

    def run_n_steps(self, n):
        steps_data = []
        # Take n steps
        for _ in range(n):
            action = self.sample_action([self.state])

            obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = get_next_state(self.state, self.img_transformer.transform(obs))

            if terminated or truncated:  # if episode ends
                print(f"Episode Reward: {self.episode_reward} - {self.name}")
                self.returns_list.append(self.episode_reward)
                self.episode_reward = 0  # reset
                if len(self.returns_list) % 100 == 0:
                    print(
                        f"\nAvg. reward (last 100 episodes): {np.mean(self.returns_list[-100:])}\n"
                    )
            else:  # if episode hasn't ended
                self.episode_reward += reward

            # Update step counter
            num_global_steps = next(self.steps_counter)

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
        assert self.state.ndim == 3  # state.shape == (84, 84, 4)

        try:
            while not coordinator.should_stop():
                # Copy weights from global value and policy models to the worker counterparts
                self.copy_global_weights()

                # Collect experience
                steps_data, num_global_steps = self.run_n_steps(steps_before_update)

                if num_global_steps >= self.max_steps:
                    coordinator.request_stop()
                    return

                # Update global value and policy models' weights using worker counterparts' gradients
                self.update(steps_data)

        except tf.errors.CancelledError:
            return

    def get_value_estimate(self, state):
        return self.worker_value_network.predict(np.array(state))

    def update(self, steps_data):
        if not steps_data[-1].done_flag:  # if the episode hasn't ended
            # Expected sum of all future rewards
            V_s_prime = self.get_value_estimate([steps_data[-1].next_state])
        else:  # if the episode has ended
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
        # Loop through steps in reverse order
        for step_data in reversed(steps_data):
            return_ = step_data.reward + self.discount_factor * V_s_prime
            advantage = return_ - self.get_value_estimate([step_data.state])
            V_s_prime = return_

            states.append(step_data.state)
            actions.append(step_data.action)
            advantages.append(advantage)
            value_targets.append(return_)

        self.update_global_weights(
            np.array(states), actions, np.array(advantages), value_targets
        )
