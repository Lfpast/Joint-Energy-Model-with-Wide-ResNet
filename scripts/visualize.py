"""Visualization utilities and CLI to run the notebook's visualization tasks."""

import numpy as np
import tensorflow as tf
from pa2.models import WRN
from pa2.data.preprocess import load_cifar10, preprocess_data
import os
from pa2.scripts.download_data import download_and_extract_cifar
from pa2.generation.energy_sampling import sampling_step, SampleBuffer, visualize_buffer_samples
from pa2.config import n_class
import matplotlib.pyplot as plt


def demo_visualize_generation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'cifar-10-batches-py')
    if not os.path.exists(data_dir):
        download_and_extract_cifar(os.path.join(base_dir, 'data'))
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    model = WRN(num_classes=n_class)
    model.build(input_shape=(None, 32, 32, 3))
    # load weights if present
    # load weights if present (from pa2/models directory)
    models_dir = os.path.join(base_dir, 'models')
    weights_path = os.path.join(models_dir, "model-20.weights.h5")
    try:
        model.load_weights(weights_path)
    except Exception:
        print(f"Could not load weights '{weights_path}' (not present). Using untrained model.")

    num_samples = 16
    x = tf.random.uniform((num_samples, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)
    for i in range(20):
        x = sampling_step(model, x, step_size=1, noise_map=0.01)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    samples = x.numpy()
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i])
            ax.axis('off')

    plt.suptitle('Generated Samples')
    plt.show()


if __name__ == '__main__':
    demo_visualize_generation()
