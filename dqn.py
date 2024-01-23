import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import Huber


class DQN:
    def __init__(self, num_actions, inp_shape, lr=1e-5):
        model = Sequential()
        model.add(Input(shape=inp_shape))
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation="relu"))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=num_actions, activation="linear"))

        self.model = model
        self.num_actions = num_actions
        self.optimizer = Adam(learning_rate=lr)
        self.huber_loss = Huber()

    def loss_fn(self, action_probs, actions, targets):
        # print(f"action_probs.shape: {action_probs.shape}")
        estimated_qvals = tf.reduce_sum(
            tf.multiply(action_probs, tf.one_hot(actions, depth=self.num_actions)),
            axis=1,
        )
        # print(f"estimated_qvals.shape: {estimated_qvals.shape}")
        # print(targets, estimated_qvals)
        loss = self.huber_loss(targets, estimated_qvals)
        # print(f"loss.shape: {loss.shape}")
        # print(f"loss: {loss}")
        return loss

    def update(self, states, actions, targets):
        with tf.GradientTape() as t:
            action_probs = self.predict(states)
            loss = self.loss_fn(action_probs, actions, targets)
        self.optimizer.minimize(
            loss=loss, var_list=self.model.trainable_weights, tape=t
        )
        return loss

    def predict(self, states):
        return self.model(states)  # returns p(a| s)

    def sample_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(state))

    def copy_weights(self, base_model):
        self.model.set_weights(base_model.get_weights())


# dqn1 = DQN(4)
# dqn2 = DQN(4)
# dqn1.copy_weights(dqn2.model)
# import gym

# env = gym.make(
#     id="ALE/Breakout-v5",
#     full_action_space=False,
#     repeat_action_probability=0.1,
#     obs_type="rgb",
# )
# obs, _ = env.reset()
# print(dqn1.sample_action(np.expand_dims(obs, axis=0)))
