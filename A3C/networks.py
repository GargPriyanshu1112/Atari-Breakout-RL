import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Rescaling, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop


class ActorCritic:
    def __init__(self, inp_shape=(84, 84, 4), num_actions=4, reg_const=0.01):
        inputs = Input(shape=inp_shape)
        x = Rescaling(scale=1.0 / 255)(inputs)
        x = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation="relu")(x)
        x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(units=256)(x)
        pi = Dense(units=num_actions)(x)
        v = Dense(units=1)(x)

        self.actor = Model(inputs=inputs, outputs=pi)
        self.critic = Model(inputs=inputs, outputs=v)
        self.reg_const = reg_const  # regularization constant

    def get_action_probs(self, states):
        """Returns p(a| s)"""
        return tf.nn.softmax(self.actor(states))

    def actor_loss_fn(self, action_probs, chosen_action_probs, advantages):
        # Policy loss
        Lp = -(advantages * tf.math.log(chosen_action_probs))
        # Entropy
        H = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1)
        # Regularize the policy loss (encourages exploration)
        loss = tf.reduce_sum(Lp + self.reg_const * H)
        return loss

    def get_v_estimate(self, states):
        """Return V(s)"""
        return tf.squeeze(self.critic(states))

    def critic_loss_fn(self, targets, preds):
        losses = tf.math.squared_difference(targets, preds)
        return tf.reduce_sum(losses)
