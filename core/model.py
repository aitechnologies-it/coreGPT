import keras_core as K
from keras_core import layers
from keras_core import initializers

from config import GPTConfig


class Block(layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=config.layer_norm_epsilon) # TODO: bias?
        self.ln2 = layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        mha = layers.MultiHeadAttention(num_heads=config.n_head, key_dim=config.hidden_size // config.n_head)
        self.attn = lambda x, training: mha(x, x, training=training, use_causal_mask=True)
        self.mlp = K.Sequential([
            layers.Dense(
                units=4*config.hidden_size, use_bias=True, activation="gelu",
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
                bias_initializer=initializers.Zeros(),
            ),
            layers.Dense(
                units=config.hidden_size, use_bias=True,
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
                bias_initializer=initializers.Zeros(),
            ),
            layers.Dropout(config.dropout)
        ])

    def call(self, x, training=None):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
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
        self.drop = layers.Dropout(config.dropout)
        # transformer blocks
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        # decoder head
        self.ln_f = layers.LayerNormalization(axis=-1)
        self.head = layers.Dense(
            units=config.vocab_size, use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.pos_emb = self.add_weight(
            name="positional",
            shape=(1, self.config.block_size, self.config.hidden_size),
            initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training=None):
        B, T = inputs.shape
        # embed sentence
        wte = self.tok_emb(inputs)
        wpe = self.pos_emb[:, :T, :]
        x = self.drop(wte + wpe, training=training)
        # attention
        for block in self.blocks:
            x = block(x, training=training)
        # compute logits
        x = self.ln_f(x)
        x = self.head(x)
        return x
    
    def summary(self):
        x = K.Input(shape=[self.config.block_size], batch_size=self.config.batch_size, dtype="int32")
        dummy = K.Model(inputs=x, outputs=self.call(x), name=self.name)
        return dummy.summary()
