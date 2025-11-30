import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
from tqdm import tqdm
from pa2.generation.energy_sampling import SampleBuffer, sampling_step, energy


def part2_train_step(optim: optimizers.Optimizer, model: tf.keras.Model, data: tf.Tensor, label: tf.Tensor, sb: SampleBuffer, sigma: float = 0.03) -> dict:
    batch_size = len(data)
    # preprocessing
    x, y = (data + tf.random.normal(data.shape) * sigma), tf.one_hot(label, sb.n_class, dtype=tf.float32)

    x_prime, y_prime = sb.sample_from_buffer(n_samples=batch_size)
    for i in range(20):
        x_prime = sampling_step(model, x_prime, label=y_prime, step_size=1, noise_map=0.01)

    sb.add_to_buffer(x_prime, y_prime)

    with tf.GradientTape() as g:
        x_logits = model(x, training=True)
        x_logits_prime = model(x_prime, training=True)

        ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x_logits)
        energy_x = energy(model, x)
        energy_x_prime = energy(model, x_prime)

        loss = tf.reduce_mean(ce_loss) + tf.reduce_mean(energy_x) - tf.reduce_mean(energy_x_prime)

    ce_loss = tf.reduce_mean(ce_loss)
    gradients = g.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    return {"loss": loss.numpy(), "class_loss": ce_loss.numpy()}


# train loop

import os


def train_loop_2(model, optimizer, train_step, train_dataset, test_dataset, epochs: int = 5, save_interval: int = 1, buffer: SampleBuffer | None = None):
    if buffer is None:
        buffer = SampleBuffer(train_dataset.element_spec[0].shape[1:], n_class=10)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0

        with tqdm(train_dataset, unit="batch") as tepoch:
            for step, (batch_x, batch_y) in enumerate(tepoch):
                loss_dict = train_step(optimizer, model, batch_x, batch_y, buffer)
                epoch_loss += loss_dict["loss"]
                num_batches += 1

                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(**loss_dict)

        from pa2.eval.eval import evaluate_accuracy
        test_accuracy = evaluate_accuracy(model, test_dataset)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        if epoch % save_interval == 0:
            # Save into pa2/models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(models_dir, exist_ok=True)
            model.save_weights(os.path.join(models_dir, f'JEM-{epoch}.weights.h5'))

    return buffer
