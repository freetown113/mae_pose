import functools
from typing import Any, NamedTuple, Callable
import os
from operator import getitem

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import pickle
import os


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
 
 leaves, treedef = jax.tree_flatten(tree_struct)
 with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
   flat_state = [np.load(f) for _ in leaves]

 return jax.tree_unflatten(treedef, flat_state)


class PatchEncoder(hk.Module):
    def __init__(
        self,
        patch_dim,
        enc_projection_dim,
        mask_proportion,
        downstream=False,
        **kwargs,
    ):
        super().__init__(name=kwargs['name']+self.__class__.__name__)
        self.patch_dim = patch_dim
        self.projection_dim = enc_projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        self.mask_token = hk.get_parameter(name='masking', 
                                     shape=(1, patch_dim), 
                                     dtype=np.float32, 
                                     init=hk.initializers.TruncatedNormal(stddev=0.02)
                                     #init=hk.initializers.VarianceScaling(distribution="normal")
                                     )
        
        self.builded = False

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape
       
        self.projection = hk.Linear(self.projection_dim)

        self.position_embedding = hk.Embed(self.num_patches, self.projection_dim, 
                                           w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                           #w_init=hk.initializers.VarianceScaling(distribution="uniform"),
                                           lookup_style="ARRAY_INDEX")

        self.num_mask = int(self.mask_proportion * self.num_patches)

    def __call__(self, patches, mask_indices, unmask_indices):
        if not self.builded:
           self.build(patches.shape)
           self.builded = True
        # Get the positional embeddings.
        batch_size = patches.shape[0]
        positions = np.arange(start=0, stop=self.num_patches, step=1)

        pos_embeddings = self.position_embedding(positions)

        pos_embeddings = jnp.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        pad_unmask_indices = unmask_indices != -1
        pad_mask_indices = mask_indices != -1

        unmasked_embeddings = jax.vmap(getitem)(patch_embeddings, unmask_indices)
        # unmasked_embeddings = jnp.take_along_axis(
        #     patch_embeddings, jnp.expand_dims(unmask_indices, axis=-1), axis=1
        # )   # (B, unmask_numbers, projection_dim)

        unmasked_positions = jax.vmap(getitem)(pos_embeddings, unmask_indices)
        # unmasked_positions = jnp.take_along_axis(
        #     pos_embeddings, jnp.expand_dims(unmask_indices, axis=-1), axis=1
        # )# (B, unmask_numbers, projection_dim)

        masked_positions = jax.vmap(getitem)(pos_embeddings, mask_indices)
        # masked_positions = jnp.take_along_axis(
        #     pos_embeddings, jnp.expand_dims(mask_indices, axis=-1), axis=1
        # )   # (B, mask_numbers, projection_dim)

        mask_tokens = jnp.tile(jnp.expand_dims(self.mask_token, axis=0), (batch_size, self.num_mask, 1))
        # mask_tokens = jnp.repeat(self.mask_token, repeats=self.num_mask, axis=0)
        # mask_tokens = jnp.repeat(
        #     jnp.expand_dims(mask_tokens, axis=0), repeats=batch_size, axis=0
        # )

        # Get the masked embeddings for the tokens.
        masked_embeddings = self.projection(mask_tokens) + masked_positions

        return (
            unmasked_embeddings,  # Input to the encoder.
            masked_embeddings,  # First part of input to the decoder.
            unmasked_positions,  # Added to the encoder outputs.
            mask_indices,  # The indices that were masked.
            unmask_indices,  # The indices that were unmaksed.
            pad_unmask_indices, 
            pad_mask_indices
        )

    # def get_random_indices(self, orig_length):
    #     mask_indices = []
    #     unmask_indices = []
    #     masked_max = int(self.num_patches * self.mask_proportion)
    #     unmasked_max = int(self.num_patches * (1 - self.mask_proportion))
    #     self.num_mask = masked_max
    #     for length in orig_length:
    #         rand_indices = np.argsort(
    #             np.random.uniform(size=length), axis=-1
    #         )
    #         num_mask = int(self.mask_proportion * length)
    #         mask_indices.append(jax.lax.pad(rand_indices[:num_mask], 
    #                                           padding_config=[(0, masked_max - num_mask, 0)], 
    #                                           padding_value=-1))
    #         unmask_indices.append(jax.lax.pad(rand_indices[num_mask:], 
    #                                           padding_config=[(0, unmasked_max - int(length) + num_mask, 0)], 
    #                                           padding_value=-1))

    #     return np.stack(mask_indices), np.stack(unmask_indices)
