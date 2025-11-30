import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np


def part1_train_step(optim: optimizers.Optimizer, model: tf.keras.Model, data: tf.Tensor, label: tf.Tensor, sigma: float = 0.03) -> dict:
    # preprocessing
    y_encoded = tf.one_hot(label, depth=10)
    gaussian_noise = tf.random.normal(shape=tf.shape(data), mean=0, stddev=sigma)
    x_noisy = data + gaussian_noise

    # train model
    with tf.GradientTape() as g:
        logits = model(x_noisy, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y_encoded, logits, from_logits=True)
        loss = tf.reduce_mean(loss)

    gradients = g.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    return {"loss": loss.numpy()}


# train loop
from tqdm import tqdm

import os


def train_loop_1(model, optimizer, train_step, train_dataset, test_dataset, epochs: int = 20, save_interval: int = 5):
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0

        # Wrap the training dataset with tqdm to create a progress bar
        with tqdm(train_dataset, unit="batch") as tepoch:
            for step, (batch_x, batch_y) in enumerate(tepoch):
                # Execute a train step and get the losses
                loss_dict = train_step(optimizer, model, batch_x, batch_y)
                epoch_loss += loss_dict["loss"]
                num_batches += 1

                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(**loss_dict)

        from pa2.eval.eval import evaluate_accuracy
        # Uncomment to test accuracy during training (Implement that first!)
        test_accuracy = evaluate_accuracy(model, test_dataset)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        print()

        if epoch % save_interval == 0:
            # Save into pa2/models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(models_dir, exist_ok=True)
            model.save_weights(os.path.join(models_dir, f'model-{epoch}.weights.h5'))

