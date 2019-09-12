import random
import os

from PIL import Image
from config.defaults import cfg

import tensorflow as tf

Image.LOAD_TRUNCATED_IMAGES = True


def load_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    # Add a channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(cfg.DATA.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(cfg.DATA.BATCH_SIZE)

    return train_ds, test_ds