import tensorflow as tf
from tensorflow import keras
from VGG16 import MyVGG16
from keras.datasets import cifar10
import numpy as np

using_gpu_index = 0  # 使用的 GPU 号码
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu_list[using_gpu_index],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]
        )
    except RuntimeError as e:
        print(e)
else:
    print("Got no GPUs")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

# model construction
model = MyVGG16(10,fc1=256,fc2=128,input_shape=(32, 32, 3))
model.build((None,32,32,3))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

# train
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
y_eva = model.evaluate(x_test, y_test, return_dict=True)

