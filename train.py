import jax
import jax.numpy as jnp
import numpy as np
from itertools import count
import os
import pickle

from utils import generate_masked_sequence, motion2video_3d
from optim_grad import MaskedAE
from dataset import get_loader_train_test, get_poinsts_from_joints_complex, prepare_data
from models import save, load

from absl import app
from absl import flags



FLAGS = flags.FLAGS


flags.DEFINE_string('data_path', '/capital/datasets/BEHAVE_updated', 'Location of the dataset')
flags.DEFINE_integer('seq_length', 64, 'image spatial size', lower_bound=32)
flags.DEFINE_integer('batch', 96, 'batch_size', lower_bound=1)
flags.DEFINE_integer('num_training_updates', int(1e6), 'number of steps to pass', lower_bound=1000)
flags.DEFINE_integer('skeleton_points', 17, 'size of a token', lower_bound=1)
flags.DEFINE_integer('skeleton_joints', 8, 'size of a token', lower_bound=1)
flags.DEFINE_integer('num_points', 3, 'size of a token', lower_bound=1)
flags.DEFINE_integer('complex', 3, 'size of a token', lower_bound=1)
flags.DEFINE_integer('enc_projection_dim', 128, 'latent space', lower_bound=32)
flags.DEFINE_integer('enc_num_heads', 4, 'size of a token', lower_bound=1)
flags.DEFINE_integer('enc_layers', 6, 'size of a token', lower_bound=1)
flags.DEFINE_integer('dec_projection_dim', 64, 'latent space', lower_bound=32)
flags.DEFINE_integer('dec_num_heads', 4, 'size of a token', lower_bound=1)
flags.DEFINE_integer('dec_layers', 2, 'size of a token', lower_bound=1)
flags.DEFINE_integer('max_length', 100, 'size of a token', lower_bound=30)
flags.DEFINE_integer('num_classes', 120, 'number of action classes', lower_bound=2)
flags.DEFINE_float('norm_eps', 1e-6, 'part of image square to mask')
flags.DEFINE_float('mask_proportion', 0.75, 'part of image square to mask')
flags.DEFINE_float('wd', 1e-4, 'weight_decay')
flags.DEFINE_float('lr', 3e-4, 'learning_rate')
flags.DEFINE_string('name', 'Masked_VAE', 'Model name')
flags.DEFINE_string('params_dir', 'weights', 'directory where models parameters are saved')
flags.DEFINE_string('dataset', 'ROSE', 'Name of the dataset')
flags.DEFINE_bool('var', True, 'calculate variance of the dataset')
flags.DEFINE_bool('pretrained', True, 'load pretrained parameters to the model')


def train(argv):

    if FLAGS.dataset == 'BEHAVE':
        if os.path.exists('3dcoodrs.pickle'):
            with open('3dcoodrs.pickle', 'rb') as handle:
                data, max_val, min_val = pickle.load(handle), 1.0883151292800903, -1.1227000951766968
        else:
            data, max_val, min_val = prepare_data(FLAGS.data_path, FLAGS.seq_length)
    elif FLAGS.dataset == 'ROSE':
        if os.path.exists('ROSE_H36M_dataset.pickle'):
            with open('ROSE_H36M_dataset.pickle', 'rb') as handle:
                data, max_val, min_val = pickle.load(handle)[:10000], 1., 0.
        else:
            pass # ADD function to handle ROSE dataset

    train_loader, test_loader, variance = get_loader_train_test(data, FLAGS, max_val, min_val)

    rng = jax.random.PRNGKey(33)

    arguments = {FLAGS[i].name: FLAGS[i].value for i in dir(FLAGS)}
    arguments.update({
        'enc_transformer_units': [FLAGS.enc_projection_dim*2, FLAGS.enc_projection_dim],
        'dec_transformer_units': [FLAGS.dec_projection_dim*2, FLAGS.dec_projection_dim],
        'num_patches': FLAGS.max_length * FLAGS.skeleton_joints,
        'patch_dim': FLAGS.num_points * FLAGS.complex, 
        'dec_max_length': int(FLAGS.max_length * FLAGS.skeleton_joints),
        'enc_max_length': int(FLAGS.max_length * FLAGS.skeleton_joints * (1 - FLAGS.mask_proportion)),
        'variance': variance
        })
    model = MaskedAE(arguments)
    
    states = model.init_params(rng, next(iter(train_loader))[:-1])
    if FLAGS.pretrained:
        states = load(FLAGS.params_dir)
    min_loss = np.inf
    #motion2video_3d(data[0].transpose((1,2,0)), './output/file.mp4')
    for i in count():
        train_losses, test_losses = [], []
        cls_train, cls_test = [], []

        for step, pose in enumerate(train_loader):

            # Get the embeddings and positions.
            states, loss = model.update_params(states, pose[:-1])
            #loss = jax.device_get(loss)
            train_losses.append(jax.device_get(loss["reconst_loss"]))
            cls_train.append(jax.device_get(loss["classif_loss"]))

        # train_pred = jnp.argmax(loss['logits'], axis=-1)
        # train_accuracy = jnp.mean(train_pred == pose[-2]).item()
        current_loss = np.mean(train_losses)
        if min_loss > current_loss:
            min_loss = current_loss
            # save(FLAGS.params_dir, states)

        for step, pose in enumerate(test_loader):

            # Get the embeddings and positions.
            key, _ = jax.random.split(rng, num=2)
            _, test_loss = model.forward.apply(states['params'], key, pose[:-1])
            #test_loss = jax.device_get(test_loss)
            test_losses.append(jax.device_get(test_loss["reconst_loss"]))
            cls_test.append(jax.device_get(test_loss["classif_loss"]))

        # predictions = jnp.argmax(test_loss['logits'], axis=-1)
        # class_accuracy = jnp.mean(predictions == pose[-2]).item()

        print(f'Epoch {i} REC: tr_loss: {np.mean(train_losses):.5f} | ts_loss: {np.mean(test_losses):.5f} '
            #   f'CLS: tr_loss: {np.mean(cls_train):.5f} | ts_loss: {np.mean(cls_test):.5f} '
            #   f'| tr_acc: {train_accuracy:.5f} | ts_acc: {class_accuracy:.5f}'
            )

        # idx = np.random.choice(FLAGS.batch)
        # padding = int(pose[-1][idx] // FLAGS.skeleton_joints)
        # test_loss = jax.device_get(test_loss)
        
        # masked = test_loss['patches'][idx]
        # masked = np.reshape(masked, (-1, FLAGS.skeleton_joints, FLAGS.num_points * FLAGS.complex))[:padding, ...]        

        # masked = ((masked + 1.0) / 2.0) * (max_val - min_val) + min_val
        # masked = get_poinsts_from_joints_complex(masked, FLAGS.skeleton_points, 
        #                                     padding, 
        #                                     FLAGS.skeleton_joints, 
        #                                     FLAGS.num_points)

        # outputs = test_loss['decoder_outputs'][idx]
        # outputs = np.reshape(outputs, (-1, FLAGS.skeleton_joints, FLAGS.num_points * FLAGS.complex))[:padding, ...]
        # outputs = ((outputs + 1.0) / 2.0) * (max_val - min_val) + min_val
        # outputs = get_poinsts_from_joints_complex(outputs, FLAGS.skeleton_points, 
        #                                     padding, 
        #                                     FLAGS.skeleton_joints, 
        #                                     FLAGS.num_points)
        
        # original = pose[0][idx]
        # original = np.reshape(original, (-1, FLAGS.skeleton_joints, FLAGS.num_points * FLAGS.complex))[:padding, ...]
        # original = ((original + 1.0) / 2.0) * (max_val - min_val) + min_val
        # original = get_poinsts_from_joints_complex(original, FLAGS.skeleton_points, 
        #                                     padding, 
        #                                     FLAGS.skeleton_joints, 
        #                                     FLAGS.num_points)
        # motion2video_3d(outputs, 
        #                 masked, 
        #                 original, 
        #                 FLAGS.dataset,
        #                 f'./output_BEHAVE/outputs_epoch_{i}.mp4', 
        #                 fps=10, 
        #                 unmasked_indexes=test_loss['unmask_indices'][idx]
        #                 )


if __name__ == '__main__':
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    print(local_devices)
    print(global_devices)
    
    with jax.default_device(global_devices[0]):
        app.run(train)