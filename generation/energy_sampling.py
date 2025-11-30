import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pa2.config import image_labels, n_class


# Energy function

def energy(model: tf.keras.Model, data: tf.Tensor, label: tf.Tensor | None = None):
    logits = model(data, training=False)
    if label is not None:
        # negative logit for the true class
        energy_xy = -tf.gather(logits, label, batch_dims=1, axis=1)
        return energy_xy
    else:
        energy_x = -tf.reduce_logsumexp(logits, axis=1)
        return energy_x


# Sampling step

def sampling_step(model: tf.keras.Model, data: tf.Tensor, label: tf.Tensor | None = None, step_size: float = 1, noise_map: float = 0.01):
    # This forces the gradient tape to only track gradients for the input data
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(data)
        if label is not None:
            Energy = tf.reduce_sum(energy(model, data, label))
        else:
            Energy = tf.reduce_sum(energy(model, data))

    gradient = g.gradient(Energy, data)

    data = data - step_size * gradient
    noise = tf.random.normal(shape=data.shape, mean=0, stddev=noise_map)
    data = data + noise

    data = tf.clip_by_value(data, 0, 1)
    return data


# A buffer storing past negative samples
class SampleBuffer:
    def __init__(self, sample_shape, n_class: int, max_samples: int = 10000, dtype: np.dtype = np.float32):
        self.max_samples = max_samples
        self.sample_shape = sample_shape
        self.n_class = n_class
        self.dtype = dtype
        self.buffer = []

    def add_to_buffer(self, samples, ids):
        if isinstance(samples, tf.Tensor):
            samples = samples.numpy()
        if isinstance(ids, tf.Tensor):
            ids = ids.numpy()

        for sample, id_ in zip(samples, ids):
            self.buffer.append((sample.astype(self.dtype), int(id_)))
            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def sample_from_buffer(self, n_samples: int, p_new: float = 0.05):
        if len(self.buffer) == 0:
            n_new = n_samples
        else:
            n_new = np.random.binomial(n_samples, p_new)

        noise, noise_class = [], []
        if n_new > 0:
            noise_arr = np.random.uniform(0, 1, size=(n_new, *self.sample_shape)).astype(self.dtype)
            noise_cls_arr = np.random.randint(0, self.n_class, size=(n_new, )).astype(np.int32)
            noise = [tf.convert_to_tensor(noise_arr[i]) for i in range(n_new)]
            noise_class = [tf.convert_to_tensor(noise_cls_arr[i]) for i in range(n_new)]

        replay, replay_class = [], []
        if n_new < n_samples and len(self.buffer) > 0:
            n_old = n_samples - n_new
            replace_flag = len(self.buffer) < n_old
            indices = np.random.choice(len(self.buffer), size=n_old, replace=replace_flag)

            for i in indices:
                sample_buff, label_buff = self.buffer[i]
                replay.append(tf.convert_to_tensor(sample_buff))
                replay_class.append(tf.convert_to_tensor(label_buff))

        if len(noise) + len(replay) == 0:
            # Fallback to random noise if buffer empty
            noise_arr = np.random.uniform(0, 1, size=(n_samples, *self.sample_shape)).astype(self.dtype)
            noise = [tf.convert_to_tensor(noise_arr[i]) for i in range(n_samples)]
            noise_class = [tf.convert_to_tensor(np.random.randint(0, self.n_class)) for _ in range(n_samples)]

        sample = tf.stack(list(noise) + list(replay), axis=0)
        sample_class = tf.stack(list(noise_class) + list(replay_class), axis=0)

        return sample, sample_class


def visualize_buffer_samples(buffer: SampleBuffer, num_samples: int = 16, p_new: float = 0.05):
    samples, labels = buffer.sample_from_buffer(num_samples, p_new)
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples[i].numpy())
        plt.xlabel(image_labels[labels[i].numpy()])
    plt.show()


def visualize_energy(model: tf.keras.Model, sample_index: int = 20, x_test=None, y_test=None):
    if x_test is None or y_test is None:
        raise ValueError("x_test and y_test must be provided to visualize energy")

    real_image = x_test[sample_index]
    real_label = tf.expand_dims(y_test[sample_index], 0)

    noise_image = tf.random.uniform(shape=real_image.shape, minval=0, maxval=1)
    grey_image = tf.ones_like(real_image) * 0.5

    images = [real_image, noise_image, grey_image]
    exy = []
    ex = []
    names = ['Real', 'Noise', 'Grey']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i, ax in enumerate(names):
        exy.append(energy(model, tf.expand_dims(images[i], 0), real_label).numpy()[0])
        ex.append(energy(model, tf.expand_dims(images[i], 0)).numpy()[0])
        axes[i].imshow(images[i])
        axes[i].set_title(f"{ax} E(x,y): {exy[i]:.2f}, E(x) {ex[i]:.2f}")
        axes[i].axis('off')
    plt.show()
