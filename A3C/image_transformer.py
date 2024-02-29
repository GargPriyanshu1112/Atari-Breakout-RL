import tensorflow as tf


class ImageTransformer:
    def __init__(self, h=84, w=84):
        self.h = h
        self.w = w

    def transform(self, img):
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.crop_to_bounding_box(img, 34, 0, 160, 160)
        img = tf.image.resize(
            img,
            size=(self.h, self.w),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        img = tf.squeeze(img)
        assert img.shape == (self.h, self.w)
        return img
