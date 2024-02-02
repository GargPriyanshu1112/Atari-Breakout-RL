import numpy as np

from image_transformer import ImageTransformer
from networks import get_networks
from utils import get_breakout_env, repeat_frame, get_next_state, Step
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
        # Get gradients and the corresponding weights of worker value model
        worker_vmodel_grads = self.worker_value_network.get_gradients(
            states, value_targets
        )
        worker_vmodel_trainable_weights = (
            self.worker_value_network.model.trainable_weights
        )

        # Apply gradients
        self.global_value_network.optimizer.apply_gradients(
            zip(worker_vmodel_grads, worker_vmodel_trainable_weights)
        )

        # Get gradients and the corresponding weights of worker policy model
        worker_pmodel_grads = self.worker_policy_network.get_gradients(
            states, value_targets
        )
        worker_pmodel_trainable_weights = (
            self.worker_policy_network.model.trainable_weights
        )

        # Apply gradients
        self.global_policy_network.optimizer.apply_gradients(
            zip(worker_pmodel_grads, worker_pmodel_trainable_weights)
        )

        # update weights using these gradients...
        # if loss needed, perhaps it is better to make function that does it all (with
        # its implementaion in the nets.py)
        # in the code perhaps we are taking gradiets and updating weights of the final dense layer as well

    def sample_action(self, state):
        self.worker_policy_network.get_logits(state)
        pass  # incomplete

    def get_value_estimate(self, state):
        return self.worker_value_network.predict(state)

    def run_n_steps(self, n):
        steps_data = []
        for _ in range(n):
            # Take a step
            action = self.sample_action(np.expand_dims(self.state, axis=0))
            # check where this works or should we use sample_action() ---------------------------
            obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = get_next_state(self.state, self.img_transformer.transform(obs))

            if terminated or truncated:  # if episode ends
                print(f"Episode Reward: {self.episode_reward} - {self.name}")
                self.returns_list.append(self.episode_reward)
                self.episode_reward = 0
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

    def run(self, coordinator, steps_before_update):  # try block not impl, check flow
        # Start state
        self.state = repeat_frame(self.img_transformer.transform(self.env.reset()[0]))
        assert self.state.ndim == 3  # state.shape == (84, 84, 4)

        while not coordinator.should_stop():
            # Copy weights from global value and policy models to the worker counterparts
            self.copy_global_weights()

            # Collect experience
            steps_data, num_global_steps = self.run_n_steps(steps_before_update)

            if num_global_steps >= self.max_steps:
                coordinator.request_stop()
                return  # see if threads are properly joined, cloesd------------

            # Update global networks'  weights using local networks' gradients -----
            self.update(steps_data)

    def update(self, steps_data):  # recheck the flow
        value_estimate = 0.0
        if not steps_data[-1].done_flag:  # if the episode hasn't ended
            # Get expected sum of all future rewards
            value_estimate = self.get_value_estimate(steps_data[-1].next_state)

        states = []
        actions = []
        advantages = []
        value_targets = []

        # Loop through steps in reverse order
        for step_data in reversed(steps_data):
            return_ = step_data.reward + self.discount_factor * value_estimate
            advantage = return_ - self.get_value_estimate(step_data.state)

            states.append(step_data.state)
            actions.append(step_data.action)
            advantages.append(advantage)
            value_targets.append(return_)

            value_estimate = return_

            self.update_global_weights(states, actions, advantages, value_targets)

        """
        Updates global policy and value networks using the local network's gradients.
        This function will build up the inputs and targets for our neural networks...
        For each step we need,
        - state (s) : inp into both the p and v networks
        - value target: G
        - policy loss:
            - pi(a| s)
            - advantage = G - V(s)

        In order to accumulate the toal return, we will use V_hat(s') to predict the future returns
        But we will use the actual rewards if we have them.
        E.g.
            If we have s1, s2, s3 with rewards r1,r2, r3
            Then G(s3) = r3 + V(s4)
                 G(s2) = r2 + r3 + V(s4)
                 G(s1) = r1 + r2 + r3 + V(s4)

        """
