import os

import fire
import numpy as np
os.environ["KERAS_BACKEND"] = "jax"
import keras_core as K
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers
from keras_core import callbacks
import tensorflow as tf

from config import GPTConfig
from model import GPT


def load_data(config):
    B, T, shift = config.batch_size, config.block_size, config.shift

    def get_tf_dataset(path):
        data = np.memmap(path, dtype=config.token_dtype, mode='r')
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

    train_dataset, n_step_train = get_tf_dataset(config.train_path)
    if config.do_eval:
        val_dataset, n_step_val = get_tf_dataset(config.val_path)
    else:
        val_dataset, n_step_val = None, None

    return train_dataset, val_dataset, n_step_train, n_step_val


def wandb_log(wandb, optimizer, batch, logs):
    try: # not working for jax
        logs['lr'] = optimizer.learning_rate
    except:
        logs['lr'] = 0.0
    wandb.log(logs)


def train(**kwargs):
    # --- CONFIG ---
    config = GPTConfig(**kwargs)

    K.mixed_precision.set_global_policy("mixed_bfloat16")
    if config.fixed_seed:
        K.utils.set_random_seed(1337)
        tf.config.experimental.enable_op_determinism()

    # --- WANDB ---
    if config.do_wandb_log:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

    # --- LOAD DATA ---
    train_dataset, val_dataset, n_step_train, n_step_val = load_data(config)

    # --- LOAD MODEL ---
    model = GPT(config)

    # --- PREPARE TRAINING ---
    total_steps = n_step_train * config.n_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    decay_steps = total_steps - warmup_steps
    print(f"Epoch steps: {n_step_train}. Total steps: {n_step_train * config.n_epoch}. "
          f"Warmup steps: {warmup_steps}. Decay steps: {decay_steps}.")

    if config.do_lr_decay:
        init_lr = config.lr / warmup_steps
        learning_rate = optimizers.schedules.CosineDecay(
            initial_learning_rate=init_lr,
            warmup_target=config.lr,
            warmup_steps=warmup_steps,
            alpha=config.min_lr,
            decay_steps=decay_steps
        )
    else:
        learning_rate = config.lr

    optimizer = optimizers.AdamW(learning_rate=learning_rate,
                                 weight_decay=config.weight_decay,
                                 beta_1=config.beta1,
                                 beta_2=config.beta2,
                                 global_clipnorm=config.grad_clip)

    model.build(input_shape=(config.batch_size, config.block_size))
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics.SparseCategoricalAccuracy(name='acc')],
        jit_compile=True
    )
    if config.verbose > 10:
        model.summary()

    my_callbacks = []
    if config.do_wandb_log:
        my_callbacks.append(
            callbacks.LambdaCallback(on_batch_end=lambda batch, logs: wandb_log(wandb, optimizer, batch, logs))
        )

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
        callbacks=[my_callbacks],
        verbose=1
    )

    # print(history.history)
    # print(model.evaluate(val_dataset, steps=n_batch_val))

    # --- SAVE ---
    if config.do_save_model:
        os.makedirs(config.out_dir, exist_ok=True)
        model.save(os.path.join(config.out_dir, f"{config.out_name}.keras"))

    return model, history, config


def main(**kwargs):  # Fire function cannot return anything.
    train(**kwargs)  # I do this to make train() return the model (eg. for when it's run in a notebook)

if __name__ == "__main__":
    fire.Fire(main)