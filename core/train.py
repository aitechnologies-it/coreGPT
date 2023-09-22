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

    def get_tf_dataset(path, token_dtype):
        data = np.memmap(path, dtype=token_dtype, mode='r')
        # First block is of size (T+1), considering +1 for the target y.
        # Plus all the shifted blocks, for which we count how many batches remaining (B-1), times the shift size
        n_step = len(data) // (T+1 + (B-1)*shift)

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
        dataset = (
            tf.data.Dataset
            .zip((x, y))
            .batch(batch_size=config.batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return dataset, n_step

    B, T, shift = config.batch_size, config.block_size, config.shift

    train_dataset, n_step_train = get_tf_dataset(config.train_path, config.token_dtype)
    if config.do_eval:
        val_dataset, n_step_val = get_tf_dataset(config.val_path, config.token_dtype)
    else:
        val_dataset, n_step_val = None, None

    return train_dataset, val_dataset, n_step_train, n_step_val


def train():
    # --- CONFIG ---
    config = GPTConfig()

    K.mixed_precision.set_global_policy("mixed_float16")
    if config.fixed_seed:
        K.utils.set_random_seed(1337)
        tf.config.experimental.enable_op_determinism()

    # --- LOAD DATA ---
    train_dataset, val_dataset, n_step_train, n_step_val = load_data(config)

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
        steps_per_epoch=n_step_train,
        epochs=config.n_epoch,
        validation_data=val_dataset,
        validation_steps=n_step_val,
        verbose=1
    )

    # print(history.history)
    # print(model.evaluate(val_dataset, steps=n_batch_val))

    os.makedirs(config.out_dir, exist_ok=True)
    model.save(os.path.join(config.out_dir, f"{config.out_name}.keras"))

    return model


if __name__ == "__main__":
    train()