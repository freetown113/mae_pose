import numpy as np
import os
import sys 
import io
import cv2
import imageio
import matplotlib.pyplot as plt
import re
import pickle

user_name = 'user'
save_pickle = './'
save_npy_path = './'
save_video = 'D:/TEST2/result/nturgbd_skeletons_s001_to_s017/output_videos'
load_txt_path = '/capital/datasets/ROSE_ready/nturgb+d_skeletons'
missing_file_path = '/capital/datasets/ROSE_ready/ntu_rgb120_missings.txt'
step_ranges = list(range(0,100)) # just parse range, for the purpose of paralle running. 


toolbar_width = 50
def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=False, save_depthxy=False):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    if not 101 > nframe > 20:
        return None
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints 
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7]
    # prune the abundant bodys 
    for each in range(max_body):
        if not len(bodymat['nbodys']):
            # prepared = []
            # for key in bodymat.keys():
            #     if 'skel_body' in key:
            #         prepared.append(key)
            # for el in prepared:
            #     del bodymat[el]
            return None
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
            return bodymat['skel_body0']


def convert_to_H36M(x):
    y = np.zeros((x.shape[0], 17, 3))
    y[:,0,:] = x[:,0,:]
    y[:,1,:] = x[:,16,:]
    y[:,2,:] = x[:,17,:]
    y[:,3,:] = x[:,18,:]
    y[:,4,:] = x[:,12,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,14,:]
    y[:,7,:] = x[:,1,:]
    y[:,8,:] = x[:,20,:]
    y[:,9,:] = x[:,2,:]
    y[:,10,:] = x[:,3,:]
    y[:,11,:] = x[:,4,:]
    y[:,12,:] = x[:,5,:]
    y[:,13,:] = x[:,6,:]
    y[:,14,:] = x[:,8,:]
    y[:,15,:] = x[:,9,:]
    y[:,16,:] = x[:,10,:]
    return y
        

def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img


def motion2video_3d(motion, save_path, fps=7, keep_imgs = False):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    # size = (608, 456)
    size = (5000, )
    motion *= (min(size) / 2.0)
    motion = motion + np.expand_dims(np.array((100, 450, 400)), axis=[0,-1])
    vlen = motion.shape[-1]

    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in range(vlen):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        ax.view_init(elev=12., azim=80)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            else:
                ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
    videowriter.close()



if __name__ == '__main__':
    missing_files = _load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path)
    # alread_exist = os.listdir(save_npy_path)
    # alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))
    data = list()
    stat = dict()
    max_lenght = 0.
    for ind, each in enumerate(datalist):
        # _print_toolbar(ind * 1.0 / len(datalist),
        #                '({:>5}/{:<5})'.format(
        #                    ind + 1, len(datalist)
        #                ))
        S = int(each[1:4])
        if S not in step_ranges:
            continue 
        # if each+'.skeleton.npy' in alread_exist_dict:
        #     print('file already existed !')
        #     continue
        if each[:20] in missing_files:
            print('file missing')
            continue 
        loadname = os.path.join(load_txt_path, each)
        mat = _read_skeleton(loadname)
        if mat is None:
            #print(f'for {each} there is no skeleton')
            continue
        mat = np.array(mat)
        mat = convert_to_H36M(mat)
        mat[...,1] = mat[...,1] * -1 + 1
        mat[...,2] = mat[...,2] - 1
        mat[...,0] = mat[...,0] + 0.75
        mat = (mat - mat.min()) / (mat.max() - mat.min())

        class_ = re.search('A\d+', os.path.splitext(each)[0]).group(0)
        #print(class_, mat.shape)
        data.append((mat, class_))
        if not ind % 1000:
            print(len(data), max_lenght, end='\r')
        if mat.shape[0] > max_lenght:
            max_lenght = mat.shape[0]
        # try:
        #     stat[mat.shape[0]]
        # except:
        #     stat[mat.shape[0]] = [mat]
        # else:
        #     stat[mat.shape[0]].append(mat)

        # motion2video_3d(mat.transpose((1,2,0)), os.path.join(save_video, each+'.mp4'), fps=20)
        # save_path = save_npy_path+'{}.npy'.format(each)
        # np.save(save_path, mat)

    for id in stat.keys():
        print(f'There {len(stat[id])} samples of length {id}')

    with open(os.path.join(save_pickle, 'ROSE_H36M_dataset.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_pickle, 'ROSE_H36M_dataset.pickle'), 'rb') as handle:
        restored = pickle.load(handle)
    
    assert all([True for i, j in zip(data, restored) if np.array_equal(i[0], j[0])]), True
    assert all([True for i, j in zip(data, restored) if i[1] == j[1]]), True
    print(len(data), max_lenght)

