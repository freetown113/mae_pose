import functools
from typing import Any, NamedTuple, Callable
import os

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

    def __call__(self, patches):
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

        mask_indices, unmask_indices = self.get_random_indices(batch_size)

        unmasked_embeddings = jnp.take_along_axis(
            patch_embeddings, jnp.expand_dims(unmask_indices, axis=-1), axis=1
        )   # (B, unmask_numbers, projection_dim)

        unmasked_positions = jnp.take_along_axis(
            pos_embeddings, jnp.expand_dims(unmask_indices, axis=-1), axis=1
        )# (B, unmask_numbers, projection_dim)

        masked_positions = jnp.take_along_axis(
            pos_embeddings, jnp.expand_dims(mask_indices, axis=-1), axis=1
        )   # (B, mask_numbers, projection_dim)

        mask_tokens = jnp.repeat(self.mask_token, repeats=self.num_mask, axis=0)

        mask_tokens = jnp.repeat(
            jnp.expand_dims(mask_tokens, axis=0), repeats=batch_size, axis=0
        )

        # Get the masked embeddings for the tokens.
        masked_embeddings = self.projection(mask_tokens) + masked_positions

        return (
            unmasked_embeddings,  # Input to the encoder.
            masked_embeddings,  # First part of input to the decoder.
            unmasked_positions,  # Added to the encoder outputs.
            mask_indices,  # The indices that were masked.
            unmask_indices,  # The indices that were unmaksed.
        )

    def get_random_indices(self, batch_size):
        rand_indices = np.argsort(
            np.random.uniform(size=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices
