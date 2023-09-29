import math

import numpy as np
import keras_core as K

from config import GPTConfig


class EmbeddingDecoder(K.layers.Layer):
    """Reimplementation of K.layers.Dense layer, but with tied_weights from the 
    work token embedding layer."""
    def __init__(
        self,
        tied_to,
        units=None,
        activation=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tied_to = tied_to
        self.units = units
        self.activation = K.activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = K.initializers.get(bias_initializer)
        self.bias_regularizer = K.regularizers.get(bias_regularizer)
        self.bias_constraint = K.constraints.get(bias_constraint)

    def build(self, input_shape):
        B, T, H = input_shape
        V = self.units
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(1, T, V),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )

    def call(self, inputs):
        w = self.tied_to.embeddings
        kernel = K.ops.transpose(w)
        x = K.ops.matmul(inputs, kernel)
        if self.use_bias:
            x = x + self.bias
        if self.activation:
            x = self.activation(x)
        return x


class CausalSelfAttention(K.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        assert config.hidden_size % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.attn = K.layers.Dense(
            units=config.hidden_size * 3, use_bias=config.bias,
            kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02),
            bias_initializer=K.initializers.Zeros(),
        )
        # regularization
        self.resid_drop = K.layers.Dropout(config.dropout)
        # output projection (special scaled init to the residual projections, per GPT-2 paper)
        self.proj = K.layers.Dense(
            units=config.hidden_size, use_bias=config.bias,
            kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02 / math.sqrt(2 * config.n_layer)),
            bias_initializer=K.initializers.Zeros(),
        )
        # Enable flash attention if backend is torch
        if config.do_flash_attention and config.backend == "torch":
            import torch
            self.flash = getattr(torch.nn.functional, 'scaled_dot_product_attention', None)
        else:
            self.flash = None
            self.attn_drop = K.layers.Dropout(config.dropout)

    def call(self, inputs, training=None):
        B, T, C = inputs.shape # batch_size, block_size, hidden_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        attended = self.attn(inputs)
        splitted = K.ops.split(attended, 3, axis=2)
        q, k, v = splitted[0], splitted[1], splitted[2]
        q = K.ops.reshape(q, (B, -1, self.config.n_head, C // self.config.n_head))
        q = K.ops.transpose(q, (0, 2, 1, 3))
        k = K.ops.reshape(k, (B, -1, self.config.n_head, C // self.config.n_head))
        k = K.ops.transpose(k, (0, 2, 1, 3))
        v = K.ops.reshape(v, (B, -1, self.config.n_head, C // self.config.n_head))
        v = K.ops.transpose(v, (0, 2, 1, 3))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash: # pytorch
            y = self.flash(q, k, v, attn_mask=None, is_causal=True,
                           dropout_p=self.config.dropout if self.training else 0)
        else:
            y = self.attention(q, k, v, training)
        y = K.ops.transpose(y, (0, 2, 1, 3)) # (B, nh, T, hs) -> (B, T, nh, hs)
        y = K.ops.reshape(y, (B, -1, C)) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y), training=training)
        return y

    def attention(self, q, k, v, training):
        att = K.ops.matmul(q, K.ops.transpose(k, (0, 1, 3, 2)))
        att = att * K.ops.rsqrt(K.ops.cast(k.shape[-1], att.dtype)) # (B, nh, T, T)
        att = self.causal_masking(att)
        att = K.ops.softmax(att, axis=-1)
        att = self.attn_drop(att, training=training)
        y = K.ops.matmul(att, v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return y

    def causal_masking(self, scores):
        _, _, Tdest, Tsrc = K.ops.shape(scores) # Tdest == Tsrc
        # Creates a lower triangular mask, so position i cannot attend to positions j>i.
        # This prevents the flow of information from the future into the past.
        mask = K.ops.tril(K.ops.ones(shape=(1, 1, Tdest, Tsrc), dtype="int32"), k=0)
        # padding positions should not contribute to attention distribution
        padding_mask = K.ops.logical_not(mask)
        # else: assume bfloat16 or float32, which have the same range
        max_value = 65504.0 if scores.dtype == "float16" else 3.38e38
        inf_mask = K.ops.cast(max_value, scores.dtype) * K.ops.cast(padding_mask, dtype=scores.dtype)
        return scores - inf_mask


class Block(K.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = K.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.ln_2 = K.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.cs_attn = CausalSelfAttention(config)
        self.mlp = K.Sequential([
            K.layers.Dense(
                units=4*config.hidden_size, use_bias=config.bias, activation="gelu",
                kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02),
                bias_initializer=K.initializers.Zeros(),
            ),
            K.layers.Dense(
                units=config.hidden_size, use_bias=config.bias,
                kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02),
                bias_initializer=K.initializers.Zeros(),
            ),
            K.layers.Dropout(config.dropout)
        ], name="mlp")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        x = x + self.cs_attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x
    

class GPT(K.Model):
    def __init__(self, config: GPTConfig, **kwargs):
        super().__init__(name="coreGPT", **kwargs)
        self.config = config

        # input embedding
        self.tok_emb = K.layers.Embedding(
            input_dim=config.vocab_size, output_dim=config.hidden_size,
            embeddings_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02),
            name="embedding",
        )
        self.drop = K.layers.Dropout(config.dropout)
        # transformer blocks
        self.blocks = K.Sequential(
            [Block(config) for _ in range(config.n_layer)],
            name="transformer_blocks",
        )
        # decoder head
        self.ln_f = K.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, axis=-1) # TODO bias
        self.head = EmbeddingDecoder(tied_to=self.tok_emb, units=config.vocab_size, use_bias=config.bias)

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.config.block_size, self.config.hidden_size),
            initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
            name="positional"
        )

    def call(self, inputs, training=None):
        B, T = inputs.shape
        # embed sentence
        wte = self.tok_emb(inputs)
        wpe = self.pos_emb[:, :T, :]
        x = self.drop(wte + wpe, training=training)
        # attention
        x = self.blocks(x)
        # compute logits
        x = self.ln_f(x)
        x = self.head(x)
        return x
    
    def summary(self):
        x = K.Input(shape=[self.config.block_size], batch_size=self.config.batch_size, dtype=self.config.token_dtype_k)
        dummy = K.Model(inputs=x, outputs=self.call(x), name=self.name)
        return dummy.summary()
    
    def get_list_exclude_from_weight_decay(self):
        to_exclude = [self.ln_f]
        for block in self.blocks.layers:
            to_exclude.append(block.ln_1)
            to_exclude.append(block.ln_2)
            for dense in block.mlp.layers:
                if hasattr(dense, "bias"):
                    to_exclude.append(dense.bias)
            if hasattr(block.cs_attn.attn, "bias"):
                to_exclude.append(block.cs_attn.attn.bias)
            if hasattr(block.cs_attn.proj, "bias"):
                to_exclude.append(block.cs_attn.proj.bias)


    def generate(self, input_ids, max_length, temperature=1.0, sample=False, top_k=None):
        if not isinstance(input_ids, np.ndarra):
            raise ValueError(f'Input input_ids should be np.ndarray, found {type(input_ids)}')
        if input_ids.ndim < 1 or input_ids.ndim > 2:
            raise ValueError(f'Input input_ids should have 1 or 2 dims, found {input_ids.ndim} dimensions.')
            
        if input_ids.ndim == 2 and input_ids.shape[0] > 1:
            raise ValueError('Input input_ids should only contain one sequence, ie. should have batch of size 1.')
        if input_ids.ndim == 1:
            input_ids = K.ops.expand_dims(input_ids, axis=0)

        for _ in range(max_length):
            _, T = input_ids.shape # sequence length
            if T >= self.config.block_size: # crop context if needed
                input_ids = input_ids[:, :T]
            logits = self(input_ids, training=False)
            logits = K.ops.divide(logits[:, -1, :], temperature)
            if top_k is not None:
                # optionally crop probabilities to only the top k options
                v, _ = K.ops.top_k(logits, top_k, sorted=True)
                logits = K.ops.identity(logits).numpy()
                logits[logits < v.numpy()[:, [-1]]] = -float('Inf')
            probabilities = K.activations.softmax(logits, axis=-1)
            if sample:
                chunk_id = K.random.categorical(K.ops.log(probabilities), num_samples=1)
            else:
                _, chunk_id = K.ops.top_k(probabilities, k=1)
            input_ids = K.ops.concatenate([
                input_ids, K.ops.reshape(K.ops.cast(chunk_id, dtype=input_ids.dtype), new_shape=(1, 1))], axis=-1
            )
        return input_ids
