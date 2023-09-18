from dataclasses import dataclass


@dataclass
class GPTConfig:
    # GPT configs
    block_size: int = 64 # 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4 # 12
    n_head: int = 12
    hidden_size: int = 256 # 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    layer_norm_epsilon: float = 1e-05

    # Train configs
    n_epoch = 2
    batch_size = 16
    weight_decay = 1e-01
    beta1 = 0.9
    beta2 = 0.95
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 100 # 2000 # how many steps to warm up for
    verbose = 100 # 10

config = GPTConfig()