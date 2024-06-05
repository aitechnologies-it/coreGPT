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
    do_flash_attention: bool = True # Only PyTorch. Warning: cannot load same model with different backend anymore

    # Train configs
    n_epoch: int = 2
    batch_size: int = 32    

    lr: float = 6e-4
    do_lr_decay: bool = True # whether to decay the learning rate
    warmup_ratio: float = 0.1
    min_lr: float = 6e-5
    weight_decay: float = 1e-1 # TODO: not all weights should
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    
    verbose: int = 1 # 10
    out_dir: str = "out"
    out_name: str = "final_model"
    backend: str = "jax"
    fixed_seed: bool = False
    do_eval_epoch: bool = True # Regular keras fit() eval (every epoch)
    do_eval_every: int = 0 # Set 0 to disable. Warning: makes each training step 50/100% slower
    do_save_model: bool = True

    do_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16" # bfloat16, float16

    do_wandb: bool = False
    wandb_project: str = "coregpt"
    wandb_run_name: str = "test1"

    # Data configs
    dataset_name: str = "shakespeare"
    shift: int = 1
    data_dir: str = "./data"
    dataset_framework: str = "tensorflow" # torch, tensorflow
    # data config specific tf.dataset
    buffer_size: int = 10000


    def __post_init__(self):
        if self.backend == "torch":
            self.token_dtype_np = np.int32 # torch doesn't support uint16 and the data max_vocab_id doesn't fit in int16
            self.token_dtype_k = "int32"
            self.train_path = os.path.join(self.data_dir, self.dataset_name, "train_int32.bin")
            self.val_path = os.path.join(self.data_dir, self.dataset_name, "val_int32.bin")
        else:
            self.token_dtype_np = np.uint16
            self.token_dtype_k = "uint16"
            self.train_path = os.path.join(self.data_dir, self.dataset_name, "train.bin")
            self.val_path = os.path.join(self.data_dir, self.dataset_name, "val.bin")
        