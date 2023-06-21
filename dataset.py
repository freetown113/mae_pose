import numpy as np
import torch
import torchvision.transforms as trfs


class Transforms:
    def __init__(self, ):
        pass


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
        
        joints = np.zeros((self.seq_length, len(idxs), 3, self.pts), dtype=np.float32)
        for i in range(joints.shape[1]):
            joints[:, i, 0, :] = patch[:, idxs[i][0], :]
            joints[:, i, 1, :] = patch[:, idxs[i][1], :]
            joints[:, i, 2, :] = patch[:, idxs[i][2], :]

        points = get_poinsts_from_joints_complex(joints, self.skeleton, self.seq_length, joints.shape[1], self.pts).transpose((2,0,1))
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


def collate_fn(input):
    out = np.stack(input)
    return np.reshape(out, (out.shape[0], np.prod(out.shape[1:3]), np.prod(out.shape[3:])))


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
    for idx in data.keys():
        if idx not in idxs:
            data_train[cnt] = data[idx]
            cnt += 1

    traindata = Dataset(data_train, 
                      cfg.num_points, 
                      cfg.skeleton_points, 
                      cfg.seq_length, 
                      min_val, 
                      max_val, 
                      transforms=transf)
    variance = traindata.compute_variance(True)

    testdata = Dataset(data_test, 
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