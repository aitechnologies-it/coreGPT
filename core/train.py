import os

import fire
os.environ["KERAS_BACKEND"] = "jax"
import keras_core as K
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers
from keras_core import callbacks

from config import GPTConfig
from model import GPT
from dataset import load_data


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
        import tensorflow as tf
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