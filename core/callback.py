import keras_core as K
try:
    import wandb
except: # Don't crash if wandb is not available, not a problem as long as config.do_wandb is False
    pass


class AddLRCallback(K.callbacks.Callback):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def on_batch_end(self, batch, logs=None):
        try:
            logs['lr'] = self.optimizer.learning_rate
        except: # not working for jax backend
            logs['lr'] = 0.0


class EvaluateCallback(K.callbacks.Callback):
    def __init__(self, config, val_dataset, n_step_val):
        self.config = config
        self.val_dataset = val_dataset
        self.n_step_val = n_step_val

    def on_batch_end(self, batch, logs=None):
        if batch % self.config.do_eval_every == 0:
            loss, accuracy = self.model.evaluate(
                self.val_dataset,
                batch_size=self.config.batch_size,
                steps=self.n_step_val,
                verbose=0
            )
            logs["val_loss"] = loss
            logs["val_acc"] = accuracy


class WandbCallback(K.callbacks.Callback):
    def __init__(self):
        self.epoch = 1

    def on_batch_end(self, batch, logs=None):
        wandb.log(logs, step=self.epoch*batch)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1