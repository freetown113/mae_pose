import jax
import numpy as np
from itertools import count
import os
import json
import pickle

from utils import generate_masked_sequence, motion2video_3d
from optim_grad import MaskedAE
from dataset import get_loader_train_test, get_poinsts_from_joints_complex

from absl import app
from absl import flags



FLAGS = flags.FLAGS


flags.DEFINE_string('data_path', '/capital/datasets/BEHAVE_updated', 'Location of the dataset')
flags.DEFINE_integer('seq_length', 64, 'image spatial size', lower_bound=32)
flags.DEFINE_integer('batch', 128, 'batch_size', lower_bound=1)
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
flags.DEFINE_float('norm_eps', 1e-6, 'part of image square to mask')
flags.DEFINE_float('mask_proportion', 0.75, 'part of image square to mask')
flags.DEFINE_float('wd', 1e-4, 'weight_decay')
flags.DEFINE_float('lr', 3e-4, 'learning_rate')
flags.DEFINE_string('name', 'Masked_VAE', 'Model name')
flags.DEFINE_bool('var', True, 'calculate variance of the dataset')


def prepare_data(inputpath, sequece_length):       
    data = dict()
    max_val = 0.0
    min_val = 0.0
    cnt = 0
    for root, dirs, files in os.walk(inputpath):
        try:
            name, ext = os.path.splitext(files[0])
        except:
            continue
        else:
            if ext not in ['.json'] or not '3dcoords' in  root:
                #print('The files in folder are not images!')
                continue

        for f in files:
            path_json = os.path.join(root, f)

            with open(path_json, "r") as read_file:
                results = json.load(read_file)
                size = len(results) // sequece_length
                if not size:
                    print(f'Warning: number of frames in the folder {root} is not compatible ', 
                          f'with the number of frames in json ({sequece_length} / {len(results)}) ',
                          f'this pair will be excluded from dataset')
                pass
                max_val = np.array([np.array(results).max(), max_val]).max()
                min_val = np.array([np.array(results).min(), min_val]).min()
                start = 0
                for i in range(size):
                    data[cnt] = results[start:start+sequece_length]
                    start += sequece_length
                    cnt +=1
        print(f'Source {root} containing {len(files)} files was handled. The dataset size has {cnt} examples now')

    with open('3dcoodrs.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('3dcoodrs.pickle', 'rb') as handle:
        restored = pickle.load(handle)

    assert data == restored, True

    return data, max_val, min_val


def train(argv):

    if os.path.exists('3dcoodrs.pickle'):
        with open('3dcoodrs.pickle', 'rb') as handle:
            data, max_val, min_val = pickle.load(handle), 1.0883151292800903, -1.1227000951766968
    else:
        data, max_val, min_val = prepare_data(FLAGS.data_path, FLAGS.seq_length)

    train_loader, test_loader, variance = get_loader_train_test(data, FLAGS, max_val, min_val)

    rng = jax.random.PRNGKey(33)

    arguments = {FLAGS[i].name: FLAGS[i].value for i in dir(FLAGS)}
    arguments.update({
        'enc_transformer_units': [FLAGS.enc_projection_dim*2, FLAGS.enc_projection_dim],
        'dec_transformer_units': [FLAGS.dec_projection_dim*2, FLAGS.dec_projection_dim],
        'num_patches': FLAGS.seq_length * FLAGS.skeleton_joints,
        'patch_dim': FLAGS.num_points * FLAGS.complex, 
        'variance': variance
        })
    model = MaskedAE(arguments)
    
    states = model.init_params(rng, next(iter(train_loader)))
    #motion2video_3d(data[0].transpose((1,2,0)), './output/file.mp4')
    for i in count():
        train_losses = []
        test_losses = []

        for step, pose in enumerate(train_loader):

            # Get the embeddings and positions.
            states, loss = model.update_params(states, pose)
            loss = jax.device_get(loss)
            train_losses.append(loss["total_loss"])

        for step, pose in enumerate(test_loader):

            # Get the embeddings and positions.
            key, _ = jax.random.split(rng, num=2)
            _, test_loss = model.forward.apply(states['params'], key, pose)
            test_loss = jax.device_get(test_loss)
            test_losses.append(test_loss["total_loss"])

        print(f'After epoch {i} Train Losses are: total={np.mean(train_losses):.5f} Test Losses are: {np.mean(test_losses):.5f}')


        # masked, idx = generate_masked_sequence(losses['patches'], losses['unmask_indices'])
        idx = np.random.choice(test_loss['patches'].shape[0])
        masked = test_loss['patches'][idx]

        masked = ((masked + 1.0) / 2.0) * (max_val - min_val) + min_val
        masked = get_poinsts_from_joints_complex(masked, FLAGS.skeleton_points, 
                                            FLAGS.seq_length, 
                                            FLAGS.skeleton_joints, 
                                            FLAGS.num_points)
        outputs = ((test_loss['decoder_outputs'][idx] + 1.0) / 2.0) * (max_val - min_val) + min_val
        outputs = get_poinsts_from_joints_complex(outputs, FLAGS.skeleton_points, 
                                            FLAGS.seq_length, 
                                            FLAGS.skeleton_joints, 
                                            FLAGS.num_points)
        original = ((pose[idx] + 1.0) / 2.0) * (max_val - min_val) + min_val
        original = get_poinsts_from_joints_complex(original, FLAGS.skeleton_points, 
                                            FLAGS.seq_length, 
                                            FLAGS.skeleton_joints, 
                                            FLAGS.num_points)
        motion2video_3d(outputs, 
                        masked, 
                        original, 
                        f'./output_complex/outputs_epoch_{i}.mp4', 
                        fps=7, 
                        unmasked_indexes=test_loss['unmask_indices'][idx]
                        )


if __name__ == '__main__':
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    print(local_devices)
    print(global_devices)
    
    with jax.default_device(global_devices[0]):
        app.run(train)