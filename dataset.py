import numpy as np


def load_data(config):
    if config.dataset_framework == "torch":
        print("Using PyTorch Dataloader.")
        return _load_data_pt(config)
    else:
        print("Using Tensorflow Dataset.")
        return _load_data_tf(config)


def _load_data_tf(config):
    import tensorflow as tf

    def get_dataset(path, training):
        data = np.memmap(path, dtype=config.token_dtype_np, mode='r')
        n_step = _compute_n_step(data, config)

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
        dataset = tf.data.Dataset.zip((x, y))

        if training:
            dataset = dataset.shuffle(buffer_size=config.buffer_size, reshuffle_each_iteration=True)

        dataset = (
            dataset
            .repeat()
            .batch(batch_size=config.batch_size,
                   drop_remainder=True,
                   num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return dataset, n_step

    train_dataset, n_step_train = get_dataset(config.train_path, training=True)
    if config.do_eval_epoch or config.do_eval_every:
        val_dataset, n_step_val = get_dataset(config.val_path, training=False)
    else:
        val_dataset, n_step_val = None, None

    return train_dataset, val_dataset, n_step_train, n_step_val


def _load_data_pt(config):
    import torch

    class MyIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, path, config):
            super().__init__()
            self.data = np.memmap(path, dtype=config.token_dtype_np, mode='r')
            self.config = config
            self.n_step = _compute_n_step(self.data, config)

        def get_streaming(self):
            S = config.shift
            while True:
                ix = torch.randint(len(self.data) - self.config.block_size, (1,))
                x = torch.from_numpy((self.data[ix:ix+self.config.block_size]).astype(np.int64))
                y = torch.from_numpy((self.data[ix+S:ix+S+self.config.block_size]).astype(np.int64))
                yield x, y
        
        def __iter__(self):
            return iter(self.get_streaming())
        
        def __len__(self):
            return len(self.data)
    
    def get_dataset(path):
        dataset = MyIterableDataset(path, config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        return dataloader, dataset.n_step

    train_dataset, n_step_train = get_dataset(config.train_path)
    if config.do_eval_epoch or config.do_eval_every:
        val_dataset, n_step_val = get_dataset(config.val_path)
    else:
        val_dataset, n_step_val = None, None

    return train_dataset, val_dataset, n_step_train, n_step_val


def _compute_n_step(data, config):
    # First block is of size (T+1), considering +1 for the target y.
    # Plus all the shifted blocks, for which we count how many batches remaining (B-1), times the shift size
    B, T, S = config.batch_size, config.block_size, config.shift
    n_step = len(data) // (T+1 + (B-1)*S)
    return n_step