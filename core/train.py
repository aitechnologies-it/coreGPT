import os
from typing import Union

import fire
import numpy as np
os.environ["KERAS_BACKEND"] = "jax"
import keras_core as K
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers
import tensorflow as tf

from config import GPTConfig
from model import GPT


def load_data(config):
    train_data = np.memmap(config.train_path, dtype=config.token_dtype, mode='r')
    val_data = np.memmap(config.val_path, dtype=config.token_dtype, mode='r')

    n_batch_train = (len(train_data)-config.block_size)//config.batch_size
    n_batch_val = (len(val_data)-config.block_size)//config.batch_size

    def get_windowed_tf_dataset(data: Union[np.memmap, np.array]):
        x = (
            tf.data.Dataset.from_tensor_slices(data[:-1])
            .window(config.block_size, shift=config.shift, stride=1, drop_remainder=True)
            .flat_map(lambda x: x.batch(config.block_size))
        )
        y = (
            tf.data.Dataset.from_tensor_slices(data[1:])
            .window(config.block_size, shift=config.shift, stride=1, drop_remainder=True)
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


def train():
    # --- CONFIG ---
    config = GPTConfig()

    if config.fixed_seed:
        K.utils.set_random_seed(1337)
    # --- LOAD DATA ---
    train_dataset, val_dataset, n_batch_train, n_batch_val = \
        load_data(config)

    # --- LOAD MODEL ---
    model = GPT(config)
    model.build(input_shape=(config.batch_size, config.block_size))
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=6e-4, weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics.SparseCategoricalAccuracy(name='acc')],
        jit_compile=True
    )
    if config.verbose > 10:
        model.summary()

    # --- BUILD --- Only needed for torch
    if config.backend == "torch":
        inp = next(iter(train_dataset))[0]
        _ = model(inp)

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

    os.makedirs(config.out_dir, exist_ok=True)
    model.save(os.path.join(config.out_dir, f"{config.out_name}.keras"))


if __name__ == "__main__":
    train()