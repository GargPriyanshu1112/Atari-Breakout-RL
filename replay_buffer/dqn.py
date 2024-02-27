import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Rescaling, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import Huber


class DQN:
    def __init__(self, inp_shape=(84, 84, 4), num_actions=4, lr=1e-5):
        model = Sequential()
        model.add(Input(shape=inp_shape))
        model.add(Rescaling(scale=1.0 / 255))
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

    def predict(self, states):
        return self.model(states)

    def compute_loss(self, preds, actions, target_Qvals):
        # print(f"action_probs.shape: {action_probs.shape}")
        pred_Qvals = tf.reduce_sum(
            preds * tf.one_hot(actions, depth=self.num_actions),
            axis=1,
        )
        # print(f"estimated_qvals.shape: {estimated_qvals.shape}")
        # print(targets, estimated_qvals)
        loss = self.huber_loss(target_Qvals, pred_Qvals)
        # print(f"loss.shape: {loss.shape}")
        # print(f"loss: {loss}")
        return loss

    def update(self, states, actions, target_Qvals):
        with tf.GradientTape() as t:
            preds = self.predict(states)
            loss = self.compute_loss(preds, actions, target_Qvals)
        # Derive gradients
        gradients = t.gradient(loss, self.model.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss

    def sample_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.expand_dims(state, axis=0)))

    def copy_weights(self, m):
        self.model.set_weights(m.get_weights())
