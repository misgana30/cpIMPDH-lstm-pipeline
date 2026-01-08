from __future__ import annotations
from typing import Optional
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def compile_model(model, learning_rate: float = 1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def make_callbacks(out_model_path: str, patience: int = 5):
    return [
        ModelCheckpoint(out_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
    ]

def fit(model, x, y, batch_size: int = 256, epochs: int = 30, validation_split: float = 0.1,
        out_model_path: str = "best_model.h5", patience: int = 5):
    cb = make_callbacks(out_model_path=out_model_path, patience=patience)
    hist = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=cb)
    return hist
