import tensorflow as tf
from tensorflow.keras import layers, models


class ResBlock(tf.keras.Model):
    def __init__(self, in_channels: int, out_channels: int, downsampling: bool = False):
        super(ResBlock, self).__init__()
        self.downsampling = downsampling
        self.stride = 2 if downsampling else 1
        self.ConvBlock = models.Sequential([
            layers.LeakyReLU(),
            layers.Conv2D(out_channels, (3, 3), strides=1, padding='same', use_bias=True),
            layers.LeakyReLU(),
            layers.Conv2D(out_channels, (3, 3), strides=self.stride, padding='same', use_bias=True)
        ])

        if in_channels != out_channels:
            self.skip = layers.Conv2D(out_channels, (1, 1), strides=self.stride, use_bias=False)
        else:
            self.skip = lambda x: x

    def call(self, x, training=False):
        Fx = self.ConvBlock(x, training=training)
        return Fx + self.skip(x)


class WRN(tf.keras.Model):
    def __init__(self, num_classes: int = 10):
        super(WRN, self).__init__()
        self.structures = models.Sequential([
            layers.Conv2D(16, (3, 3), strides=1, padding='same', use_bias=True),
            ResBlock(16, 64, downsampling=False),
            ResBlock(64, 64, downsampling=False),
            ResBlock(64, 128, downsampling=True),
            ResBlock(128, 128, downsampling=False),
            ResBlock(128, 256, downsampling=True),
            ResBlock(256, 256, downsampling=False),
            layers.LeakyReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes)
        ])

    def call(self, x, training=False):
        return self.structures(x, training=training)

    def build(self, input_shape):
        self.structures.build(input_shape)
        super(WRN, self).build(input_shape)
