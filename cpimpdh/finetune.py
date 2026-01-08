
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


@dataclass
class FineTuneResult:
    model: tf.keras.Model
    history: Optional[object] = None


def finetune_full_unfreeze(
    pretrained: tf.keras.Model,
    X_ft,
    y_ft,
    lr: float = 3e-4,
    decay_steps: int = 2000,
    decay_rate: float = 0.2,
    epochs: int = 8,
    batch_size: int = 64,
    val_split: float = 0.1,
) -> FineTuneResult:
    model = tf.keras.models.clone_model(pretrained)
    model.set_weights(pretrained.get_weights())
    for l in model.layers:
        l.trainable = True

    schedule = ExponentialDecay(lr, decay_steps, decay_rate, staircase=True)
    model.compile(optimizer=Adam(schedule), loss="sparse_categorical_crossentropy")
    hist = model.fit(X_ft, y_ft, epochs=epochs, batch_size=batch_size, validation_split=val_split)
    return FineTuneResult(model=model, history=hist)


def finetune_discriminative_lr(
    pretrained: tf.keras.Model,
    X_ft,
    y_ft,
    low_layer_count: int = 3,
    lr_low: float = 1e-5,
    lr_high: float = 3e-4,
    epochs: int = 6,
    batch_size: int = 64,
) -> FineTuneResult:
    model = tf.keras.models.clone_model(pretrained)
    model.set_weights(pretrained.get_weights())
    for l in model.layers:
        l.trainable = True

    low_layers = model.layers[:low_layer_count]
    high_layers = model.layers[low_layer_count:]
    opt_low, opt_high = Adam(lr_low), Adam(lr_high)

    vars_low = sum([list(l.trainable_variables) for l in low_layers], [])
    vars_high = sum([list(l.trainable_variables) for l in high_layers], [])
    n_low = len(vars_low)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def disc_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(loss_fn(y, logits))
        grads = tape.gradient(loss, model.trainable_variables)
        opt_low.apply_gradients(zip(grads[:n_low], vars_low))
        opt_high.apply_gradients(zip(grads[n_low:], vars_high))
        return loss

    # simple loop (no validation) to keep identical behavior
    n = X_ft.shape[0]
    for _ in range(epochs):
        for i in range(0, n, batch_size):
            disc_step(X_ft[i:i+batch_size], y_ft[i:i+batch_size])

    return FineTuneResult(model=model, history=None)
