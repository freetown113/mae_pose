import haiku as hk
import jax
import optax
import jax.numpy as jnp
import functools

from patching import PatchEncoder
from models import Encoder, Decoder, MLP


def create_forward_fn(fns, args):
   def fwd_pass(patches):

        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = fns['PatchEncoder'](**args)(patches)

        encoder_outputs = fns['Encoder'](**args)(unmasked_embeddings)

        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = jnp.concatenate([encoder_outputs, masked_embeddings], axis=1)

        decoder_outputs = fns['Decoder'](**args)(decoder_inputs)

        masked_input = jnp.take_along_axis(patches, jnp.expand_dims(mask_indices, axis=-1), axis=1)
        reconstruced_masked_input = jnp.take_along_axis(decoder_outputs, jnp.expand_dims(mask_indices, axis=-1), axis=1)

        total_loss = jnp.mean(jnp.square(masked_input - reconstruced_masked_input)) / args['variance']

        return total_loss, dict({
            'total_loss': total_loss,
            'decoder_outputs': decoder_outputs,
            'unmask_indices': unmask_indices,
            'patches': patches
        })

   return hk.transform(fwd_pass)


class MaskedAE:
    def __init__(self, arguments):
        self.args = arguments
        
    def init_params(self, rng, dummy_input):
        key, sub = jax.random.split(rng, num=2)

        self.forward = create_forward_fn(dict({
            'Encoder': Encoder,
            'Decoder': Decoder,
            'PatchEncoder': PatchEncoder
        }), self.args)
        params = self.forward.init(key, dummy_input)

        self.optim = optax.adamw(learning_rate=self.args['lr'], weight_decay=self.args['wd'])
        optim_params = self.optim.init(params)

        states = dict({
            'params': params,
            'optim': optim_params,
            'key': sub
        })

        return states

    @functools.partial(jax.jit, static_argnums=0)
    def update_params(self, states, patches):
        key, sub = jax.random.split(states['key'], num=2)
        (loss, model_output), grads = jax.value_and_grad(self.forward.apply, has_aux=True)(states['params'], key, patches)

        updates, opt_state = self.optim.update(grads, states['optim'], states['params'])
        params = optax.apply_updates(states['params'], updates)

        states = dict({
            'params': params,
            'optim': opt_state,
            'key': sub
        })

        return states, model_output