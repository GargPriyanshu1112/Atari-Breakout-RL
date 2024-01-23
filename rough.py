import numpy as np
import keras
from keras.losses import Huber
import tensorflow as tf


huber_loss = Huber(reduction=tf.keras.losses.Reduction.NONE)

a = [[0, 1], [0, 0]]
b = [[0.6, 0.4], [0.4, 0.6]]

# a = np.array([[1, 2]])
# print(a.shape)
# b = np.array([[1, 1]])
# print(b.shape)

print(huber_loss(a, b))

print(tf.compat.v1.losses.huber_loss(a, b))
