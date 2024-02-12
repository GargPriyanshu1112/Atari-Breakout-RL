import tensorflow as tf
import tensorflow_probability as tfp
from keras.models import Sequential, Model
from keras.layers import Input, Rescaling, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop


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
        self.optimizer = RMSprop(0.00025, 0.99, 0.0, 1e-6)
        self.optimizer.build(self.model.trainable_weights)

    def predict(self, states):
        return tf.squeeze(self.model(states))

    def loss_fn(self, targets, preds):
        losses = tf.math.squared_difference(targets, preds)
        return tf.reduce_sum(losses)


class PolicyNetwork:
    def __init__(self, inp_shape, num_actions, shared_layers, reg_const=0.01):
        inputs = Input(shape=inp_shape)
        x = shared_layers(inputs)
        outputs = Dense(units=num_actions)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.reg_const = reg_const  # regularization constant
        self.optimizer = RMSprop(0.00025, 0.99, 0.0, 1e-6)
        self.optimizer.build(self.model.trainable_weights)

    def get_logits(self, states):
        return self.model(states)

    def get_probs(self, states):
        return tf.nn.softmax(self.get_logits(states))

    def sample_action(self, state):
        logits = self.get_logits(state)
        distibution = tfp.distributions.Categorical(logits=logits)  # READ
        action = distibution.sample()[0]
        return action

    def loss_fn(self, action_probs, selected_action_probs, advantages):
        C = self.reg_const
        H = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1)
        Lp = -(advantages * tf.math.log(selected_action_probs))
        loss = tf.reduce_sum(Lp + C * H)
        return loss


def get_networks(inp_shape, num_actions):
    shared_layers = get_shared_layers()
    value_network = ValueNetwork(inp_shape, shared_layers)
    policy_network = PolicyNetwork(inp_shape, num_actions, shared_layers)
    return value_network, policy_network
