import jax
import haiku as hk
import jax.numpy as jnp

class MLP(hk.Module):
    def __init__(self, dropout_rate, hidden_units, name=None):
        super().__init__(name=name)
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units

    def __call__(self, x):
        for out in self.hidden_units:
            x = hk.Linear(out)(x)
            x = jax.nn.gelu(x)
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        return x
    

class Encoder(hk.Module):
    def __init__(self, enc_num_heads, 
                 enc_layers, 
                 enc_projection_dim, 
                 enc_transformer_units, 
                 norm_eps, 
                 name=None,
                 **kwargs):
        super().__init__(name=name+self.__class__.__name__)
        self.num_heads = enc_num_heads
        self.num_layers = enc_layers
        self.proj_dim = enc_projection_dim
        self.transf_units = enc_transformer_units
        self.eps = norm_eps
        
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            attention_output = hk.MultiHeadAttention(num_heads=self.num_heads,
                                                     key_size=self.proj_dim,
                                                     model_size=self.proj_dim,
                                                     w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                                     #w_init=hk.initializers.RandomNormal(),
                                                     name='EncMha')(x, x, x)

            x = attention_output + x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            x_m = MLP(hidden_units=self.transf_units, dropout_rate=0.1)(x)
            x = x + x_m

        outputs = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
        return outputs
    

class Decoder(hk.Module):
    def __init__(self, dec_layers, 
                 dec_num_heads, 
                 patch_dim,
                 dec_projection_dim, 
                 dec_transformer_units, 
                 norm_eps, 
                 num_patches,
                 name=None,
                 **kwargs
                 ):
        super().__init__(name=name+self.__class__.__name__)
        self.num_heads = dec_num_heads
        self.num_layers = dec_layers
        self.patch_dim = patch_dim
        self.proj_dim = dec_projection_dim
        self.transf_units = dec_transformer_units
        self.eps = norm_eps
        self.num_patches = num_patches
        
    def __call__(self, x):
        x = hk.Linear(self.proj_dim)(x)

        for _ in range(self.num_layers):
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            attention_output = hk.MultiHeadAttention(num_heads=self.num_heads,
                                                     key_size=self.proj_dim,
                                                     model_size=self.proj_dim,
                                                     #w_init=hk.initializers.RandomNormal(), 
                                                     w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                                     name='DecMha')(x, x, x)
            
            x = attention_output + x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            x_m = MLP(hidden_units=self.transf_units, dropout_rate=0.1, name='DecMlp')(x)
            x = x + x_m

        x = x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
        x = hk.Flatten(name='DecFlat')(x)
        logits = hk.Linear(self.num_patches*self.patch_dim)(x)
        predict = jax.nn.tanh(logits)
        predict = jnp.reshape(predict, (-1, self.num_patches, self.patch_dim))

        return predict