import tensorflow as tf


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
