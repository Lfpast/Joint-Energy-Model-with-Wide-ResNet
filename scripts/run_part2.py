"""Simple script to run Part 2 (JEM) training with sample buffer."""

from pa2.data.preprocess import load_cifar10, preprocess_data, build_datasets
import os
from pa2.scripts.download_data import download_and_extract_cifar
from pa2.models import WRN
from pa2.config import learning_rate, default_batch_size, n_class
from pa2.train.train_part2 import train_loop_2, part2_train_step
from pa2.generation.energy_sampling import SampleBuffer
from pa2.eval.eval import show_misclassified
from tensorflow.keras import optimizers


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'cifar-10-batches-py')
    if not os.path.exists(data_dir):
        download_and_extract_cifar(os.path.join(base_dir, 'data'))
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    train_dataset, test_dataset = build_datasets(X_train_p, Y_train_p, X_test_p, Y_test_p, batch_size=default_batch_size)

    model = WRN(num_classes=n_class)
    model.build(input_shape=(None, 32, 32, 3))

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    buffer = SampleBuffer(sample_shape=X_train_p[0].shape, n_class=n_class)

    buf = train_loop_2(model, optimizer, part2_train_step, train_dataset, test_dataset, epochs=2, save_interval=1, buffer=buffer)

    show_misclassified(model, test_dataset)


if __name__ == '__main__':
    main()
