import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import visualizer
import math
import matplotlib.pyplot as plt

# translation matrix (translate along z-axis)
trans_t = lambda t: torch.tensor(
    [[1, 0, 0, 0], 
     [0, 1, 0, 0], 
     [0, 0, 1, t], 
     [0, 0, 0, 1]], dtype=torch.float32,
)

# rotation matrix (phi, rotate x-axis)
rot_phi = lambda phi: torch.tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ],
    dtype=torch.float32,
)

# rotation matrix (theta, rotate y-axis)
rot_theta = lambda th: torch.tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ],
    dtype=torch.float32,
)

def pose_spherical(theta, phi, radius):
    """
    define camera to world transformation matrix according to camera pose
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    #? exchange y & z axis, flip x axis?
    c2w = (
        torch.tensor(
            [         
                [-1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        ) @ c2w
    return c2w


def load_blender_data_2(basedir, half_res=False, testskip=1, visdir=None):
    all_imgs = []
    all_poses = []
    meta = []
    # counts = [0]
    with open(os.path.join(basedir, "transforms.json"), "r") as fp:
        meta = json.load(fp)
    imgs = []
    poses = []
    
    for frame in meta["frames"]:
        fname = os.path.join(basedir, frame["file_path"])
        try:
            with open(fname) as f:
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame["transform_matrix"]))
        except FileNotFoundError:
            print(f"file {fname} not found")
    imgs = (np.array(imgs) / 255.0).astype(np.float32)  
    alpha = np.ones((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    imgs = np.concatenate((imgs, alpha), axis=3)# keep all 4 channels (RGBA)
    print("imges shape", imgs.shape)
    print("alpha shape", alpha.shape)

    poses = np.array(poses).astype(np.float32)
    # counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)
    total = len(imgs)
    
    train_ratio = 0.7
    train_upper_idx = math.floor(train_ratio * total)
    test_ratio = 0.2
    test_upper_idx = train_upper_idx + math.floor(test_ratio * total)
    validation_upper_idx = total - 1
    print(f"idx {total} {math.floor(train_ratio * total)} {train_upper_idx} {test_upper_idx} {validation_upper_idx}" )
    # i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    i_split = [np.arange(0, train_upper_idx),  
                np.arange(test_upper_idx + 1, validation_upper_idx),
                np.arange(train_upper_idx + 1, test_upper_idx)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)


    render_poses = torch.stack(
        [
            pose_spherical(angle, -15.0, 5.0)
            for angle in np.linspace(75, 165, 40 + 1)[:-1]
        ],
        0,
    )

    render_poses_0 = torch.stack(
        [    
            pose_spherical(angle, 15, 5.0)
            for angle in np.linspace(75, 165, 40 + 1)[:-1]
        ],
        0,
    )
    
    render_poses = torch.concatenate(
        [render_poses, render_poses_0],
        0
    )

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        plt.imshow(imgs[4])

    # visualize
    # min_d = poses[:, :3, 3].min(axis=0)
    # max_d = poses[:, :3, 3].max(axis=0)
    # print(f"min: {min_d} max: {max_d}")

    # vis = visualizer.CameraPoseVisualizer([min_d[0] * 2., max_d[0] * 2.], [min_d[1] * 2., max_d[1] * 2.], [min_d[2] * 2., max_d[2] * 2.])
    # plist = [poses, render_poses]
    # # plist = [render_poses]
    # # plist = [poses]

    # clist = ['k', 'c']
    # # c2w_44 = np.concatenate([c2w[:, :-1], np.array([[0, 0, 0, 1]])], 0)
    # for l, c in zip(plist, clist):
    #     plen = l.shape[0]
    #     print(plen)
    #     hwf = l[0,:3,-1]
    #     for idx, p in enumerate(l[::2]):
    #         #print("p shape", p[:,:-1].shape)
    #         # print("prepare p ", p.shape)
    #         right = p[:, 0]
    #         up = p[:, 1]
    #         eye = - p[:, 2]
    #         pos = p[:, 3]
    #         c2w = np.stack([right, up, eye, pos], 1)
    #         # print(c2w)
    #         # p = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], 0)
    #         # print(p)
    #         # print(hwf)
    #         # vis.extrinsic2pyramid(p, plt.cm.rainbow(idx % plen), 0.1, aspect_ratio=hwf[0]/hwf[1])
    #         vis.extrinsic2pyramid(c2w, c, 1, aspect_ratio=0.75)

    # os.makedirs(visdir, exist_ok=True)
    # path = os.path.join(visdir, "camera_pose.jpg")
    # vis.show()
    # vis.save(path)
    # print("i_split", i_split)
    return imgs, poses, render_poses, [H, W, focal], i_split