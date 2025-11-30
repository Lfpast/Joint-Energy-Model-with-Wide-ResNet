import tensorflow as tf
from tensorflow.keras import layers, models


class ResBlockRE(tf.keras.Model):
    def __init__(self, in_channels: int, out_channels: int, downsampling: bool = False):
        super(ResBlockRE, self).__init__()
        self.downsampling = downsampling
        self.stride = 2 if downsampling else 1
        self.ConvBlock = models.Sequential([
            layers.Dropout(0.1),
            layers.LeakyReLU(),
            layers.Conv2D(out_channels, (3, 3), strides=1, padding='same', use_bias=True),
            layers.Dropout(0.1),
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


class WRNRE(tf.keras.Model):
    def __init__(self, num_classes: int = 10):
        super(WRNRE, self).__init__()
        self.structures = models.Sequential([
            layers.Conv2D(16, (3, 3), strides=1, padding='same', use_bias=True),
            ResBlockRE(16, 64, downsampling=False),
            ResBlockRE(64, 64, downsampling=False),
            ResBlockRE(64, 128, downsampling=True),
            ResBlockRE(128, 128, downsampling=False),
            ResBlockRE(128, 256, downsampling=True),
            ResBlockRE(256, 256, downsampling=False),
            layers.Dropout(0.1),
            layers.LeakyReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes)
        ])

    def call(self, x, training=False):
        return self.structures(x, training=training)
