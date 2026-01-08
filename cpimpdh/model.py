from __future__ import annotations
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization, TimeDistributed

def build_pretrained_lstm(vocab_size: int, max_len: int, embedding_dim: int = 300, lstm_units: int = 256, dropout_rate: float = 0.2) -> Model:
    inputs = Input(shape=(max_len - 1,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len - 1)(inputs)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = TimeDistributed(Dense(vocab_size, activation="softmax"))(x)
    return Model(inputs, outputs)
