import keras as K
try:
    import wandb
except: # Don't crash if wandb is not available, not a problem as long as config.do_wandb is False
    pass


class AddLRCallback(K.callbacks.Callback):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def on_batch_end(self, batch, logs=None):
        logs['lr'] = self.optimizer.learning_rate


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
    def __init__(self, n_step_epoch):
        self.offset = 0
        self.batch = 0
        self.n_step_epoch = n_step_epoch

    def on_batch_end(self, batch, logs=None):
        self.batch = batch
        wandb.log(logs, step=self.offset + batch)

    def on_epoch_end(self, epoch, logs=None):
        self.offset += self.n_step_epoch

    def on_test_end(self, logs=None):
        wandb.log(
            {
                "val_loss": logs['loss'],
                "val_acc": logs['acc'],
            },
            step=self.offset + self.batch
        )