import os
import jax
import pickle
import haiku as hk
import numpy as np
import jax.numpy as jnp
from typing import Optional


def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def load(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


class SelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""

    def projection(
        self,
        x: jax.Array,
        head_size: int,
        name: Optional[str] = None,
    ) -> jax.Array:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init,
                    with_bias=self.with_bias, b_init=self.b_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    def __call__(
            self,
            query: jnp.ndarray,
            key: Optional[jnp.ndarray] = None,
            value: Optional[jnp.ndarray] = None,
            real: int = 0,
            attention_mask: Optional[jnp.ndarray] = None,
            aux_mask = None
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = self.projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = self.projection(key, self.key_size, "key")  # [T, H, K]

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        aux_mask = jnp.expand_dims(aux_mask, axis=(1,-1))
        attn_logits = jnp.where(aux_mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        if attention_mask is not None:
            mask = jnp.ones((query.shape[0], self.num_heads, real, real))
            mask = jnp.pad(mask, pad_width=((0, 0), 
                                            (0, 0),
                                            (0, query.shape[1] - real), 
                                            (0, query.shape[1] - real)), 
                            mode='constant', constant_values=0)

        return super().__call__(query, key, value, mask)
    

def cross_entropy_loss(labels, logits, num_classes):    
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    classifier_loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
    return classifier_loss


def label_smoothing(one_hot_targets, label_smoothing):
    on_value = 1.0 - label_smoothing
    num_classes = one_hot_targets.shape[-1]
    off_value = label_smoothing / num_classes
    one_hot_targets = one_hot_targets * on_value + off_value
    return one_hot_targets


def smoothed_loss(labels, logits, num_classes, smoothing_value=0.1):
    one_hot_targets = jax.nn.one_hot(labels, num_classes)
    soft_targets = label_smoothing(one_hot_targets, smoothing_value)
    loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
    return jnp.mean(loss)


def focal_loss(gamma=2., alpha=4., num_classes=None):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(labels, logits):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        """
        epsilon = 1.e-9
        logits = logits + epsilon
        one_hot_labels = jax.nn.one_hot(labels, num_classes)

        one_hot_labels = label_smoothing(one_hot_labels, 0.1)

        cross_enpropy = -one_hot_labels * jax.nn.log_softmax(logits)
        weight = jax.lax.pow((1. - jax.nn.softmax(logits)), gamma)
        fl = alpha * weight * cross_enpropy
        reduced_fl = jnp.max(fl, axis=-1)
        return jnp.mean(reduced_fl)
    return focal_loss_fixed
    

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
                 enc_max_length,
                 name=None,
                 **kwargs):
        super().__init__(name=name+self.__class__.__name__)
        self.num_heads = enc_num_heads
        self.num_layers = enc_layers
        self.proj_dim = enc_projection_dim
        self.transf_units = enc_transformer_units
        self.eps = norm_eps
        self.max_length = enc_max_length
        
    def __call__(self, x, attention_mask):
        original_shape = x.shape
        assert x.shape[1] == self.max_length, True
        # if x.shape[1] != self.max_length:
        #     x = jnp.pad(x, pad_width=((0, 0), 
        #                                 (0, self.max_length - x.shape[1]), 
        #                                 (0, 0)), 
        #                     mode='constant', constant_values=0)

        for _ in range(self.num_layers):
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            attention_output = hk.MultiHeadAttention(num_heads=self.num_heads,
                                                     key_size=self.proj_dim,
                                                     model_size=self.proj_dim,
                                                     w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                                     #w_init=hk.initializers.RandomNormal(),
                                                     name='EncMha')(x, x, x, jnp.expand_dims(attention_mask, axis=(1,-1)))
            
            # attention_output = SelfAttention(num_heads=self.num_heads,
            #                                  key_size=self.proj_dim,
            #                                  model_size=self.proj_dim,
            #                                  w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            #                                  name='EncMha')(x, real=original_shape[1], attention_mask=True, aux_mask=attention_mask)

            x = attention_output + x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            x_m = MLP(hidden_units=self.transf_units, dropout_rate=0.1)(x)
            x = x + x_m

        outputs = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
        return outputs #[:, :original_shape[1], :]
    

class Decoder(hk.Module):
    def __init__(self, dec_layers, 
                 dec_num_heads, 
                 patch_dim,
                 dec_projection_dim, 
                 dec_transformer_units, 
                 norm_eps, 
                 num_patches,
                 dec_max_length,
                 num_classes,
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
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.max_length = dec_max_length
        
    def __call__(self, x, attention_mask, labels):
        original_shape = x.shape
        assert x.shape[1] == self.max_length, True
        # if x.shape[1] != self.max_length:
        #     x = jnp.pad(x, pad_width=((0, 0), 
        #                                 (0, self.max_length - x.shape[1]), 
        #                                 (0, 0)), 
        #                     mode='constant', constant_values=0)
            
        x = hk.Linear(self.proj_dim)(x)

        for _ in range(self.num_layers):
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            attention_output = hk.MultiHeadAttention(num_heads=self.num_heads,
                                                     key_size=self.proj_dim,
                                                     model_size=self.proj_dim,
                                                     #w_init=hk.initializers.RandomNormal(), 
                                                     w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                                     name='DecMha')(x, x, x, jnp.expand_dims(attention_mask, axis=(1,-1)))
            
            # attention_output = SelfAttention(num_heads=self.num_heads,
            #                                  key_size=self.proj_dim,
            #                                  model_size=self.proj_dim,
            #                                  w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            #                                  name='EncMha')(x, real=original_shape[1], attention_mask=True)
            
            x = attention_output + x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            x_m = MLP(hidden_units=self.transf_units, dropout_rate=0.1, name='DecMlp')(x)
            x = x + x_m

        x = x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
        


        # x = x * jnp.tile(jnp.expand_dims(attention_mask, axis=-1), (1, 1, x.shape[-1]))
        x = jnp.where(jnp.expand_dims(attention_mask, axis=-1), x, 0.)
        flattened = hk.Flatten(name='DecFlat')(x)
        logits = hk.Linear(self.num_patches*self.patch_dim, 
                           name='ReconstLogits')(flattened)
        predict = jax.nn.tanh(logits)
        predict = jnp.reshape(predict, (-1, self.num_patches, self.patch_dim))
        cls_logits = hk.Linear(self.num_classes, 
                            #    b_init=hk.initializers.Constant(-np.log((1.-0.01)/0.01)),
                               name='ClassificationLogits')(flattened)  
        # classifier_loss = jnp.mean(cross_entropy_loss(labels, cls_logits, self.num_classes))
        # classifier_loss = focal_loss(num_classes=self.num_classes)(labels, cls_logits)
        classifier_loss = smoothed_loss(labels, cls_logits, self.num_classes)




        # # x = x * jnp.tile(jnp.expand_dims(attention_mask, axis=-1), (1, 1, x.shape[-1]))
        # x = jnp.where(jnp.expand_dims(attention_mask, axis=-1), x, 0.)
        
        # flattened = hk.Flatten(name='DecFlat')(x)
        # logits = hk.Linear(self.num_patches*self.patch_dim)(flattened)
        # predict = jax.nn.tanh(logits)                       
        # predict = jnp.reshape(predict, (-1, self.num_patches, self.patch_dim))

        # x = jnp.where(jnp.expand_dims(attention_mask, axis=-1), x, -1e30)
        # x = hk.Flatten(name='DecFlat2')(x)
        # cls_logits = hk.Linear(self.num_classes)(x)
        # classifier_loss = jnp.mean(cross_entropy_loss(labels, cls_logits, self.num_classes))
        
        return predict, cls_logits, classifier_loss