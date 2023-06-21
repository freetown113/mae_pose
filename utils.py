import imageio
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2

from mpl_toolkits.mplot3d import Axes3D
# import matplotlib as mpl
# print(mpl.projections.get_projection_names())



def generate_masked_sequence(patches, unmask_indices):
    # Choose a random patch and it corresponding unmask index.
    idx = np.random.choice(patches.shape[0])
    patch = patches[idx]
    unmask_index = unmask_indices[idx]

    # Build a numpy array of same shape as patch.
    new_patch = np.zeros_like(patch)

    # Iterate of the new_patch and plug the unmasked patches.
    count = 0
    for i in range(unmask_index.shape[0]):
        new_patch[unmask_index[i]] = patch[unmask_index[i]]
    return new_patch, idx


def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img


def motion2video_3d(outputs, masked, original, save_path, fps=10, unmasked_indexes=None, keep_imgs = False):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    # size = (608, 456)
    size = (608, )
    outputs *= (min(size) / 2.0)
    masked *= (min(size) / 2.0)
    original *= (min(size) / 2.0)

    outputs = outputs + np.expand_dims(np.array((100, 450, 400)), axis=[0,-1])
    masked = masked + np.expand_dims(np.array((300, 450, 400)), axis=[0,-1])
    original = original + np.expand_dims(np.array((500, 450, 400)), axis=[0,-1])
    vlen = outputs.shape[-1]

    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    colors = dict({
        'original': ["#05c5ff", "#0569ff", "#0509ff"], 
        'masked': ["#ff0505", "#ff6d05", "#ffc505"], 
        'predict': ["#b0ff05", "#33ff05", "#05ff9f"]
        })
    versions = dict({
        'predict': outputs, 
        'masked': masked, 
        'original': original
        })


    for f in range(vlen):

        fig = plt.figure(0, figsize=(30, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)

        ax.view_init(elev=12., azim=80)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            for vrs in versions.keys():
                xs, ys, zs = [np.array([versions[vrs][:,:,f][limb[0], j], versions[vrs][:,:,f][limb[1], j]]) for j in range(3)]
                if unmasked_indexes is not None and vrs == 'masked':
                    index = f * len(joint_pairs) + i
                    if index in unmasked_indexes:
                        vrs = 'original'
                    else:
                        vrs = 'masked'
                if joint_pairs[i] in joint_pairs_left:
                    ax.plot(-xs, -zs, -ys, color=colors[vrs][0], lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                elif joint_pairs[i] in joint_pairs_right:
                    ax.plot(-xs, -zs, -ys, color=colors[vrs][1], lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                else:
                    ax.plot(-xs, -zs, -ys, color=colors[vrs][2], lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
    videowriter.close()