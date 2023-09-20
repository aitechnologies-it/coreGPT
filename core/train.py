import os
from typing import Union

import fire
import numpy as np
import tensorflow as tf
os.environ["KERAS_BACKEND"] = "jax"#"tensorflow"
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers

from config import GPTConfig
from model import GPT


def load_data(config, train_path, val_path):
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

    n_batch_train = (len(train_data)-config.block_size)//config.batch_size
    n_batch_val = (len(val_data)-config.block_size)//config.batch_size

    def get_windowed_tf_dataset(data: Union[np.memmap, np.array]):
        x = (
            tf.data.Dataset.from_tensor_slices(data[:-1])
            .window(config.block_size, shift=1, stride=1, drop_remainder=True)
            .flat_map(lambda x: x.batch(config.block_size))
        )
        y = (
            tf.data.Dataset.from_tensor_slices(data[1:])
            .window(config.block_size, shift=1, stride=1, drop_remainder=True)
            .flat_map(lambda x: x.batch(config.block_size))
        )

        return (
            tf.data.Dataset
            .zip((x, y))
            .batch(batch_size=config.batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    train_dataset = get_windowed_tf_dataset(train_data)
    val_dataset = get_windowed_tf_dataset(val_data)

    return train_dataset, val_dataset, n_batch_train, n_batch_val


def get_model(config: GPTConfig):
    model = GPT(config)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=6e-4, weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    if config.verbose > 10:
        model.summary()

    return model


def train(
    data_dir: str = "../data",
):
    train_path = os.path.join(data_dir, "shakespeare/train.bin")
    val_path = os.path.join(data_dir, "shakespeare/val.bin")

    # --- CONFIG ---
    config = GPTConfig()

    # --- LOAD DATA ---
    train_dataset, val_dataset, n_batch_train, n_batch_val = \
        load_data(config, train_path, val_path)

    # --- LOAD MODEL ---
    model = GPT(config)
    model.build((config.batch_size, config.block_size))
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=6e-4, weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics.SparseCategoricalAccuracy(name='acc')],
        jit_compile=True
    )
    if config.verbose > 10:
        model.summary()

    # --- TRAIN ---
    history = model.fit(
        train_dataset,
        steps_per_epoch=n_batch_train,
        epochs=config.n_epoch,
        validation_data=val_dataset,
        validation_steps=n_batch_val,
        verbose=1
    )

    print(history.history)

    model.save("model/final_model.keras")

if __name__ == "__main__":
    fire.Fire(train)