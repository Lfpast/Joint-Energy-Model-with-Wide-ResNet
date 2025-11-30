import os
import pickle
import numpy as np
import tensorflow as tf


def load_cifar10_batch(batch_filename: str):
    with open(batch_filename, 'rb') as f:
        dict_data = pickle.load(f, encoding='bytes')
    X = dict_data[b'data']
    Y = dict_data[b'labels']
    X = X.reshape(-1, 3, 32, 32).astype("float32")
    X = np.transpose(X, (0, 2, 3, 1))  # Convert to NHWC
    Y = np.array(Y)
    return X, Y


def load_cifar10(data_dir: str | None = None):
    X_train = []
    Y_train = []
    # There are 5 training batches
    if data_dir is None:
        # Default to data inside the package
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cifar-10-batches-py')
        data_dir = os.path.abspath(data_dir)

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        X, Y = load_cifar10_batch(batch_file)
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    # Load test batch
    X_test, Y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, Y_train, X_test, Y_test


def preprocess_data(X, Y):
    X = X / 255.0  # Normalize to [0,1]
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)
    return X, Y


# Utility to build tf.data datasets
def build_datasets(X_train_p, Y_train_p, X_test_p, Y_test_p, batch_size=128):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_p, Y_train_p))
    train_dataset = train_dataset.shuffle(buffer_size=50000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_p, Y_test_p))
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset
