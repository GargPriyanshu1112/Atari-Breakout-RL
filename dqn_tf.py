# Import dependencies
import gym
import tensorflow as tf

import matplotlib.pyplot as plt


class ImageTransformer:
    def transform(self, img, IMG_SIZE=84):
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.crop_to_bounding_box(img, 35, 0, 160, 160)
        img = tf.image.resize(
            img,
            size=(IMG_SIZE, IMG_SIZE),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        img = tf.cast(img, dtype=tf.float32) / 255.0
        img = tf.squeeze(img)
        assert img.shape == (IMG_SIZE, IMG_SIZE)
        return img


if __name__ == "__main__":
    print(tf.__version__)
    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    start_state, info = env.reset()

    # print(start_state.shape)
    # print(info)

    imgT = ImageTransformer()
    imgT.transform(start_state)
