import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Rescaling, Conv2D, Flatten, Dense

"""
Policy and Value network share their parameters
"""


def shared_layers():
    return Sequential(
        [
            Rescaling(scale=1.0 / 255),
            Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation="relu"),
            Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation="relu"),
            Flatten(),
            Dense(units=256),
        ]
    )


class PolicyNetwork:
    def __init__(self, inp_shape, num_actions, reg_const=0.01):
        inputs = Input(shape=inp_shape)
        x = shared_layers()(inputs)
        outputs = Dense(units=num_actions)(x)

        self.model = Model(inputs=inputs, outputs=outputs)

    def get_logits(self, inp):
        return self.model(inp)

    def get_action_probs(self, inp):
        logits = self.get_logits(inp)
        return tf.nn.softmax(logits)

    def sample_action(self, state):
        pass


m = PolicyNetwork(inp_shape=(84, 84, 4), num_actions=4)


class ValueNetwork:
    def __init__(self, inp_shape):
        inputs = Input(shape=inp_shape)
        x = shared_layers()(inputs)
        outputs = Dense(units=1)(x)

        self.model = Model(inputs=inputs, outputs=outputs)


# class PolicyNetwork:
#     """
#     base model -|
#     final_dense with softmax. (prob for each action...)
#     sample action from logits (result before softmax...)

#     entrop-first vid...
#     """

#     pass
