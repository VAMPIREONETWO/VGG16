import tensorflow as tf
from keras import layers, Model


class MyVGG16(Model):
    def __init__(self, classes: int, pooling: str = 'max', input_shape=None,
                 fc1: int = 4096, fc2: int = 4096, dp1=0, dp2=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # extract VGG16 layers
        self.ls = [layer for layer in tf.keras.applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            classes=classes,
            input_shape=input_shape,
            pooling=pooling
        ).layers]

        # fully connected layers
        self.fc1 = FC(fc1, dp1)
        self.fc2 = FC(fc2, dp2)
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
    def __init__(self, num, dp=0, **kwargs):
        super().__init__(**kwargs)

        self.fc = layers.Dense(num)
        self.nb = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if dp == 0:
            self.dp = lambda x: x
        else:
            self.dp = layers.Dropout(dp)

    def call(self, inputs, *args, **kwargs):
        outputs = self.fc(inputs)
        outputs = self.nb(outputs)
        outputs = self.relu(outputs)
        outputs = self.dp(outputs)
        return outputs
