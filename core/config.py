import os
from dataclasses import dataclass

import numpy as np


@dataclass
class GPTConfig:
    # GPT configs
    block_size: int = 1024 # 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12 # 12
    n_head: int = 12
    hidden_size: int = 768 # 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    layer_norm_epsilon: float = 1e-05

    # Train configs
    n_epoch = 1
    batch_size = 4
    weight_decay = 1e-01
    beta1 = 0.9
    beta2 = 0.95
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # 2000 # how many steps to warm up for
    verbose = 1 # 10
    out_dir = "out"
    out_name = "final_model"
    backend = os.environ["KERAS_BACKEND"]
    fixed_seed = True

    # Data configs
    dataset_name = "shakespeare"
    shift = 1
    data_dir = "../data"

    def __post_init__(self):
        if self.backend == "torch":
            self.token_dtype = np.int32 # torch doesn't support uint16 and the data max_vocab_id doesn't fit in int16
            self.train_path = os.path.join(self.data_dir, self.dataset_name, "train_int32.bin")
            self.val_path = os.path.join(self.data_dir, self.dataset_name, "val_int32.bin")
        else:
            self.token_dtype = np.uint16
            self.train_path = os.path.join(self.data_dir, self.dataset_name, "train.bin")
            self.val_path = os.path.join(self.data_dir, self.dataset_name, "val.bin")

config = GPTConfig()