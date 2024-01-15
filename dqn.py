import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import Adam


class DQN:
    def __init__(self, inp_shape=(84, 84, 4), num_actions=4, lr=1e-5):
        model = Sequential()
        model.add(Input(shape=inp_shape))
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation="relu"))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=num_actions, activation="linear"))

        self.model = model
        self.optimizer = Adam(learning_rate=lr)
        self.cost = None
        self.num_actions = num_actions

    def copy_weights(self, base_model):
        print(base_model.get_weights())
        print("Before...")
        print(self.model.get_weights())
        print("After...")
        self.model.set_weights(base_model.get_weights())
        print(self.model.get_weights())
        print(self.model.get_weights() is base_model.get_weights())

    def sample_action(self, obs, eps):
        if np.random.random() < eps:
            return np.random.choice(self.num_actions)
        else:
            return None


dqn1 = DQN()
dqn2 = DQN()
dqn1.copy_from(dqn2.model)
