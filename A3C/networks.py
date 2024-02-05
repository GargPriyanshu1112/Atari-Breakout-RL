import tensorflow as tf
import tensorflow_probability as tfp
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
        return tf.squeeze(self.model(states))

    def loss_fn(self, targets, preds):
        losses = tf.math.squared_difference(targets, preds)
        return tf.reduce_sum(losses)

    def get_gradients(self, states, targets):
        with tf.GradientTape() as t:
            preds = self.predict(states)
            loss = self.loss_fn(preds, targets)
        # Derive gradients
        gradients = t.gradient(loss, self.model.trainable_weights)
        return gradients


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

    def sample_action(self, state):
        logits = self.get_logits(state)
        distibution = tfp.distributions.Categorical(logits=logits)  # READ
        action = distibution.sample()[0]
        return action

    def get_probs(self, states, actions):  # TODO
        return tf.nn.softmax(self.get_logits(states))

    def loss_fn(self, action_probs, selected_action_probs, advantages):  # TODO
        C = self.reg_const
        H = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1)
        Lp = -(advantages * tf.math.log(selected_action_probs))
        loss = tf.reduce_sum(Lp + C * H)
        return loss

    def get_gradients(self, states, actions, advantages):  # TODO
        with tf.GradientTape() as t:
            action_probs = self.get_probs(states, actions)  # p(a| s)
            selected_action_probs = tf.reduce_sum(
                tf.multiply(action_probs, tf.one_hot(actions, depth=4)), axis=1
            )
            # print(f"action_probs: {action_probs.shape}")
            # print(f"selected_action_probs: {selected_action_probs.shape}")
            # print(f"selected_action_probs: {selected_action_probs}")
            # print(f"advantages: {advantages.shape}")
            # print(f"advantages: {advantages}")
            loss = self.loss_fn(action_probs, selected_action_probs, advantages)
        # Derive gradients
        gradients = t.gradient(loss, self.model.trainable_weights)
        return gradients


def get_networks(inp_shape, num_actions):  # are parameters actually shared ?
    shared_layers = get_shared_layers()
    value_network = ValueNetwork(inp_shape, shared_layers)
    policy_network = PolicyNetwork(inp_shape, num_actions, shared_layers)
    return value_network, policy_network


# if __name__ == "__main__":
#     v, p = get_networks((84, 84, 4), 4)
#     print(v.model.layers[1].get_weights() is p.model.layers[1].get_weights())
#     print(v.model.layers[1].get_weights() == p.model.layers[1].get_weights())
