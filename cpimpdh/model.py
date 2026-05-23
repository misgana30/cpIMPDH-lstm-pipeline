from __future__ import annotations
import json
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization, TimeDistributed


def build_pretrained_lstm(vocab_size: int, max_len: int, embedding_dim: int = 300, lstm_units: int = 256, dropout_rate: float = 0.2) -> Model:
    inputs = Input(shape=(max_len - 1,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = TimeDistributed(Dense(vocab_size, activation="softmax"))(x)
    return Model(inputs, outputs)


def load_pretrained(path: str) -> tf.keras.Model:
    """Load a .h5 model saved by Keras 2.x, compatible with Keras 3."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except (TypeError, ValueError) as e:
        if "time_major" in str(e) or "Unrecognized keyword" in str(e):
            pass
        else:
            raise

    with h5py.File(path, "r") as f:
        config = json.loads(f.attrs["model_config"])
        batch_shape = None
        for layer in config["config"]["layers"]:
            if layer["class_name"] == "InputLayer":
                batch_shape = layer["config"]["batch_input_shape"]
                break
        if batch_shape is None:
            raise ValueError(f"Could not find InputLayer in model config: {path}")

        input_len = batch_shape[1]
        max_len = input_len + 1

        emb_shape = f["model_weights/embedding/embedding/embeddings:0"].shape
        vocab_size, embedding_dim = emb_shape
        lstm_kernel_shape = f["model_weights/lstm/lstm/lstm_cell/kernel:0"].shape
        lstm_units = lstm_kernel_shape[1] // 4

    model = build_pretrained_lstm(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
    )
    model.load_weights(path)
    return model
