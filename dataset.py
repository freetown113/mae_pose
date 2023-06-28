import numpy as np
import torch
import os
import torchvision.transforms as trfs
import jax
import pickle
import json
import jax.numpy as jnp


class Dataset:
    def __init__(self, data, pts, skeleton, seq_length, min_val, max_val, transforms=None):
        self.data = data
        self.pts = pts
        self.skeleton = skeleton
        self.transforms = transforms
        self.seq_length = seq_length
        self.min_val = min_val
        self.max_val = max_val
        self.wrap_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = self.transforms(sample)

        return sample
    
    def get_joints(self, patch):
        idxs = [[0, 1], [1, 2], [2, 3], [0, 4], 
                [4, 5], [5, 6], [0, 7], [7, 8], 
                [8, 9], [8, 11], [8, 14], [9, 10], 
                [11, 12], [12, 13], [14, 15], [15, 16]]
        
        joints = np.zeros((self.seq_length, len(idxs), 2, self.pts), dtype=np.float32)
        for i in range(joints.shape[1]):
            joints[:, i, 0, :] = patch[:, idxs[i][0], :]
            joints[:, i, 1, :] = patch[:, idxs[i][1], :]

        points = get_poinsts_from_joints(joints, self.skeleton, self.seq_length, joints.shape[1], self.pts).transpose((2,0,1))
        assert np.array_equal(patch, points), True

        return joints
    
    def get_joints_compex(self, patch):
        idxs = [[1, 2, 3], [4, 5, 6], [1, 0, 4],
                [0, 7, 8], [8, 9, 10], [11, 8, 14],
                [14, 15, 16], [11, 12, 13]]
        
        seq_length = patch.shape[0]
        joints = np.zeros((seq_length, len(idxs), 3, self.pts), dtype=np.float32)
        for i in range(joints.shape[1]):
            joints[:, i, 0, :] = patch[:, idxs[i][0], :]
            joints[:, i, 1, :] = patch[:, idxs[i][1], :]
            joints[:, i, 2, :] = patch[:, idxs[i][2], :]

        points = get_poinsts_from_joints_complex(joints, self.skeleton, seq_length, joints.shape[1], self.pts).transpose((2,0,1))
        assert np.array_equal(patch, points), True

        return joints

    def wrap_data(self):
        for k in self.data.keys():
            self.data[k] = self.get_joints_compex(np.asarray(self.data[k], dtype=np.float32))

    def compute_variance(self, tricked=False):
        if tricked:
            return 0.019715022268702257
        data = np.zeros((self.__len__(), self.seq_length, self.skeleton, self.pts))
        for i, k in enumerate(self.data.keys()):
            data[i] = self.data[k]
        return np.var((data - self.min_val) / (self.max_val - self.min_val))
    

def get_poinsts_from_joints_complex(patch, num_pts, seq_length, num_joints, pts):
    ''' idxs = [[1, 2, 3], [4, 5, 6], [1, 0, 4],
                [0, 7, 8], [8, 9, 10], [11, 8, 14],
                [14, 15, 16], [11, 12, 13]]
    '''
    patch = np.reshape(patch, (seq_length, num_joints, -1, pts))
    points = np.zeros((seq_length, num_pts, pts), dtype=np.float32)
    points[:, 0, :] = patch[:, 2, 1, :]
    points[:, 1, :] = patch[:, 0, 0, :]
    points[:, 2, :] = patch[:, 0, 1, :]
    points[:, 3, :] = patch[:, 0, 2, :]
    points[:, 4, :] = patch[:, 1, 0, :]
    points[:, 5, :] = patch[:, 1, 1, :]
    points[:, 6, :] = patch[:, 1, 2, :]
    points[:, 7, :] = patch[:, 3, 1, :]
    points[:, 8, :] = patch[:, 4, 0, :]
    points[:, 9, :] = patch[:, 4, 1, :]
    points[:, 10, :] = patch[:, 4, 2, :]
    points[:, 11, :] = patch[:, 7, 0, :]
    points[:, 12, :] = patch[:, 7, 1, :]
    points[:, 13, :] = patch[:, 7, 2, :]
    points[:, 14, :] = patch[:, 6, 0, :]
    points[:, 15, :] = patch[:, 6, 1, :]
    points[:, 16, :] = patch[:, 6, 2, :]
    return points.transpose((1,2,0))


def get_poinsts_from_joints(patch, num_pts, seq_length, num_joints, pts):
    '''    idxs = [[0, 1], [1, 2], [2, 3], [0, 4], 
            [4, 5], [5, 6], [0, 7], [7, 8], 
            [8, 9], [8, 11], [8, 14], [9, 10], 
            [11, 12], [12, 13], [14, 15], [15, 16]]
    '''
    patch = np.reshape(patch, (seq_length, num_joints, -1, pts))
    points = np.zeros((seq_length, num_pts, pts), dtype=np.float32)
    points[:, 0, :] = patch[:, 0, 0, :]
    points[:, 1, :] = patch[:, 0, 1, :]
    points[:, 2, :] = patch[:, 2, 0, :]
    points[:, 3, :] = patch[:, 2, 1, :]
    points[:, 4, :] = patch[:, 4, 0, :]
    points[:, 5, :] = patch[:, 4, 1, :]
    points[:, 6, :] = patch[:, 5, 1, :]
    points[:, 7, :] = patch[:, 7, 0, :]
    points[:, 8, :] = patch[:, 7, 1, :]
    points[:, 9, :] = patch[:, 11, 0, :]
    points[:, 10, :] = patch[:, 11, 1, :]
    points[:, 11, :] = patch[:, 12, 0, :]
    points[:, 12, :] = patch[:, 12, 1, :]
    points[:, 13, :] = patch[:, 13, 1, :]
    points[:, 14, :] = patch[:, 14, 0, :]
    points[:, 15, :] = patch[:, 14, 1, :]
    points[:, 16, :] = patch[:, 15, 1, :]
    return points.transpose((1,2,0))


class RoseCollator(object):
    def __init__(self, max_length, mask_portion, num_patches):
        self.max_length = max_length
        self.mask_proportion = mask_portion
        self.num_patches = num_patches

    def randon_sample(self, orig_length):
        mask_indices = []
        unmask_indices = []
        masked_max = int(self.num_patches * self.mask_proportion)
        unmasked_max = int(self.num_patches * (1 - self.mask_proportion))
        self.num_mask = masked_max
        for length in orig_length:
            rand_indices = np.argsort(
                np.random.uniform(size=length), axis=-1
            )
            num_mask = int(self.mask_proportion * length)
            mask_indices.append(jax.lax.pad(rand_indices[:num_mask], 
                                              padding_config=[(0, masked_max - num_mask, 0)], 
                                              padding_value=-1))
            unmask_indices.append(jax.lax.pad(rand_indices[num_mask:], 
                                              padding_config=[(0, unmasked_max - int(length) + num_mask, 0)], 
                                              padding_value=-1))

        return jnp.stack(mask_indices), jnp.stack(unmask_indices)

    def __call__(self, input):
        labels = jnp.array([ t[1] for t in input ])

        lengths = [ t[0].shape[0] * t[0].shape[1] for t in input ]
        mask_indices, unmask_indices = self.randon_sample(lengths)

        batch = [t[0] for t in input]
        padded = [jax.lax.pad(el, padding_config=[(0, self.max_length - el.shape[0], 0), 
                                                  (0, 0, 0), (0, 0, 0), (0, 0, 0)], 
                                                  padding_value=0.0) for el in batch]

        poses = jnp.stack(padded)
        poses = jnp.reshape(poses, (poses.shape[0], np.prod(poses.shape[1:3]), np.prod(poses.shape[-2:])))
        mask = (poses != 0)

        return poses, mask, mask_indices, unmask_indices, labels, np.asarray(lengths, dtype=np.uint16)


class BehaveCollator(RoseCollator):
    def __init__(self, max_length, mask_portion, num_patches):
        super().__init__(max_length, mask_portion, num_patches)
        self.labels = None

    def __call__(self, input):
        if self.labels is None:
            self.labels = np.random.randint(0, 120, size=len(input))

        lengths = [input[0].shape[0] * input[0].shape[1]] * len(input)
        mask_indices, unmask_indices = self.randon_sample(lengths)
        batch = jnp.stack(input)
        
        padded = jax.lax.pad(batch, padding_config=[(0, 0, 0), (0, self.max_length - batch.shape[1], 0), 
                                                    (0, 0, 0), (0, 0, 0), (0, 0, 0)], 
                                                    padding_value=0.0)

        poses = jnp.reshape(padded, (padded.shape[0], np.prod(padded.shape[1:3]), np.prod(padded.shape[-2:])))
        mask = (poses != 0)

        return poses, mask, mask_indices, unmask_indices, self.labels, np.asarray(lengths, dtype=np.uint16)



class DatasetRose(Dataset):
    def __init__(self, data, pts, skeleton, seq_length, min_val, max_val, transforms=None):
        self.data = data
        self.pts = pts
        self.skeleton = skeleton
        self.transforms = transforms
        self.seq_length = seq_length
        self.min_val = min_val
        self.max_val = max_val
        self.wrap_data()

    def wrap_data(self):
        data = dict()
        for key in self.data.keys():
            data[key] = tuple([super().get_joints_compex(np.asarray(self.data[key][0], dtype=np.float32)), 
                               np.array(int(self.data[key][1][1:]))])
        self.data = data

    def __getitem__(self, index):
        pose, lable = self.data[index]
        pose = self.transforms(pose)

        return pose, lable


def get_loader_train_test(data, cfg, max_val, min_val, aux_transform=False):
    print(f'Max/ min values is {max_val}/ {min_val}')
    if aux_transform:
        transf = trfs.Compose([lambda x: np.rot90(),
                               lambda x: np.fliplr(),
                               lambda x: np.flipup(),
                               lambda x: (x - min_val) / (max_val - min_val),
                               lambda x: x * 2 - 1.0
            ])
    else:
        transf = trfs.Compose([lambda x: (x - min_val) / (max_val - min_val),
                               lambda x: x * 2 - 1.0,
                               
            ])

    idxs = np.random.choice(len(data), int(len(data)*0.1), replace=False)
    data_test = {i: data[idx] for i, idx in enumerate(idxs)}
    data_train = dict()
    cnt = 0
    for idx in data.keys() if type(data) is dict else range(len(data)):
        if idx not in idxs:
            data_train[cnt] = data[idx]
            cnt += 1

    if cfg.dataset == 'ROSE':
        DatasetClass = DatasetRose
        collate_fn = RoseCollator(cfg.max_length, cfg.mask_proportion, cfg.max_length * cfg.skeleton_joints)
    elif cfg.dataset == 'BEHAVE':
        DatasetClass = Dataset
        collate_fn = BehaveCollator(cfg.max_length, cfg.mask_proportion, cfg.max_length * cfg.skeleton_joints)  # collate_array
    else:
        raise NotImplementedError(f'The dataset class is not implemented for the type you provide {cfg.dataset}')        

    traindata = DatasetClass(data_train, 
                             cfg.num_points, 
                             cfg.skeleton_points, 
                             cfg.seq_length, 
                             min_val, 
                             max_val, 
                             transforms=transf)
    variance = traindata.compute_variance(True)

    testdata = DatasetClass(data_test, 
                            cfg.num_points, 
                            cfg.skeleton_points, 
                            cfg.seq_length, 
                            min_val, 
                            max_val, 
                            transforms=transf)
    
    loader_train = torch.utils.data.DataLoader(dataset=traindata,
                                            batch_size=cfg.batch, 
                                            collate_fn=collate_fn,
                                            shuffle=True, drop_last=True)
    
    loader_test = torch.utils.data.DataLoader(dataset=testdata,
                                            batch_size=cfg.batch, 
                                            collate_fn=collate_fn,
                                            shuffle=False, drop_last=True)
    
    return loader_train, loader_test, variance



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


# def get_loader(data, cfg, max_val, min_val, aux_transform=False):
#     print(f'Max/ min values is {max_val}/ {min_val}')
#     if aux_transform:
#         transf = trfs.Compose([lambda x: np.rot90(),
#                                lambda x: np.fliplr(),
#                                lambda x: np.flipup(),
#                                lambda x: (x - min_val) / (max_val - min_val),
#                                lambda x: x * 2 - 1.0
#             ])
#     else:
#         transf = trfs.Compose([lambda x: (x - min_val) / (max_val - min_val),
#                                lambda x: x * 2 - 1.0,
                               
#             ])


#     dataset = Dataset(data, 
#                       cfg.num_points, 
#                       cfg.skeleton_points, 
#                       cfg.seq_length, 
#                       min_val, 
#                       max_val, 
#                       transforms=transf)
#     variance = dataset.compute_variance(True)
    
#     dataloader = torch.utils.data.DataLoader(dataset=dataset,
#                                             batch_size=cfg.batch, 
#                                             collate_fn=collate_fn,
#                                             shuffle=True, drop_last=True)
    
#     return dataloader, variance