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
