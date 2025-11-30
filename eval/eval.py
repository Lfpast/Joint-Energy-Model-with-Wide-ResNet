import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pa2.config import image_labels


def evaluate_accuracy(model, dataset):
    correct = 0
    total = 0
    with tqdm(dataset, unit="batch") as tepoch:
        tepoch.set_description(f"Evaluating")
        for batch_x, batch_y in tepoch:
            batch_size = tf.shape(batch_x)[0]

            logits = model(batch_x, training=False)
            preds = tf.argmax(logits, axis=1)

            batch_y = tf.cast(batch_y, tf.int64)
            correct += tf.reduce_sum(tf.cast(preds == batch_y, tf.int32)).numpy()
            total += batch_size.numpy()

            accuracy = correct / total if total != 0 else 0
            tepoch.set_postfix(accuracy=accuracy)

    return accuracy


def show_misclassified(model, dataset, samples_to_show: int = 4):
    correct = 0
    total = 0
    misclassifiedSamples = []
    accuracies = []

    with tqdm(dataset, unit="batch") as tepoch:
        tepoch.set_description(f"Evaluating")
        for batch_x, batch_y in tepoch:
            batch_size = tf.shape(batch_x)[0]

            logits = model(batch_x, training=False)
            preds = tf.argmax(logits, axis=1)

            batch_y = tf.cast(batch_y, tf.int64)
            correct += tf.reduce_sum(tf.cast(preds == batch_y, tf.int32)).numpy()
            total += batch_size.numpy()

            misclassified_idx = tf.where(preds != batch_y)
            for idx in misclassified_idx:
                img = batch_x[idx[0]].numpy()
                predLabel = preds[idx[0]].numpy()
                trueLabel = batch_y[idx[0]].numpy()
                misclassifiedSamples.append((img, predLabel, trueLabel))

            accuracy = correct / total if total != 0 else 0
            accuracies.append(accuracy)
            tepoch.set_postfix(accuracy=accuracy)

    if len(accuracies) == 0:
        finalAccuracy = 0
    else:
        finalAccuracy = np.sum(accuracies) / len(accuracies)

    if len(misclassifiedSamples) == 0:
        print("No misclassifications found in dataset.")
        print(f"Test Accuracy: {finalAccuracy * 100:.2f}%")
        return

    # Choose samples (min(samples_to_show, len(misclassifiedSamples)))
    import random
    sampled_misclassified = random.sample(misclassifiedSamples, min(samples_to_show, len(misclassifiedSamples)))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for i, (img, pred, true) in enumerate(sampled_misclassified):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {image_labels[pred]}, True: {image_labels[true]}")
        plt.axis('off')
    plt.show()

    print()
    print(f"Test Accuracy: {finalAccuracy * 100:.2f}%")
