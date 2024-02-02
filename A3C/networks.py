import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Rescaling, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop

from config import INP_SHAPE, NUM_ACTIONS


def get_shared_layers():
    return Sequential(
        [
            Rescaling(scale=1.0 / 255),
            Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation="relu"),
            Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation="relu"),
            Flatten(),
            Dense(units=256),
        ]
    )


class ValueNetwork:
    def __init__(self, inp_shape, shared_layers):
        inputs = Input(shape=inp_shape)
        x = shared_layers(inputs)
        outputs = Dense(units=1)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.optimizer = RMSprop(0.00025, 0.99, 0.0, 1e-6)  # check from paper

    def predict(self, states):
        return self.model(states)  # see dim

    def loss_fn(self, preds, targets):
        loss = None  # (squared error)..
        return loss

    def get_gradients(self, states, targets):  # change
        with tf.GradientTape() as t:
            preds = self.predict(states)
            loss = self.loss_fn(preds, targets)
        # Derive gradients
        gradients = t.gradient(loss, self.model.trainable_weights)
        return gradients
        # return zip(gradients, self.model.trainable_weights)


class PolicyNetwork:
    def __init__(self, inp_shape, num_actions, shared_layers, reg_const=0.01):
        inputs = Input(shape=inp_shape)
        x = shared_layers(inputs)
        outputs = Dense(units=num_actions)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.optimizer = RMSprop(0.00025, 0.99, 0.0, 1e-6)  # check from paper
        self.reg_const = reg_const  # regularization constant

    def get_logits(self, states):
        assert states.ndim == 4
        return self.model(states)

    def get_probs(self, states):
        return tf.nn.softmax(self.get_logits(states))

    def sample_action(self, state):
        pass

    def loss_func(self, states):
        probs = self.get_probs(states)  # p(a| s)
        entropy = probs * tf.log(probs)  # reduce sum why
        selected_action_probs = None  # ??
        advantages = None  # ??
        # Calculate loss
        loss = None  # from notes...
        return loss

    def get_gradients(self, states, actions, advantages):  # change
        with tf.GradientTape() as t:
            x = self.get_probs(states)
            loss = self.loss_fn()
        # Derive gradients
        gradients = t.gradient(loss, self.model.trainable_weights)
        return gradients
        # return zip(gradients, self.model.trainable_weights)


def get_networks(inp_shape, num_actions):  # are parameters actually shared ?
    shared_layers = get_shared_layers()
    value_network = ValueNetwork(inp_shape, shared_layers)
    policy_network = PolicyNetwork(inp_shape, num_actions, shared_layers)
    return value_network, policy_network
