import tensorflow as tf
from keras.datasets import cifar10
from tensorflow import keras
from keras import layers, Sequential, Model


class MyVGG16(Model):
    def __init__(self, classes: int, pooling: str = 'max', input_shape=None,
                 fc1: int = 4096, fc2: int = 4096,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # extract VGG16 layers
        self.ls = [layer for layer in tf.keras.applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            classes=10,
            input_shape=input_shape,
            pooling=pooling
        ).layers]

        # full connection layers
        self.fc1 = FC(fc1)
        self.fc2 = FC(fc2)
        self.fc3 = layers.Dense(classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        outputs = self.ls[0](inputs)
        for i in range(1, len(self.ls)):
            outputs = self.ls[i](outputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        return outputs


class FC(layers.Layer):
    def __init__(self, num, **kwargs):
        super().__init__(**kwargs)

        self.fc = layers.Dense(num)
        self.nb = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        outputs = self.fc(inputs)
        outputs = self.nb(outputs)
        outputs = self.relu(outputs)
        return outputs
