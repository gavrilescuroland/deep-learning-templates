import tensorflow as tf
from tensorflow.keras import layers


class Net(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = layers.Conv2D(6, 5, activation='relu', input_shape=(28, 28, 1))
        self.pool = layers.MaxPool2D(2, 2)
        self.conv2 = layers.Conv2D(16, 5, activation='relu')
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(10)

    def call(self, inputs):
        x = self.pool(self.conv1(inputs))
        x = self.pool(self.conv2(x))
        x = self.fc1(layers.Flatten()(x))
        x = self.fc2(x)
        return self.fc3(x)

