import loader
import numpy as np 
from typing import Tuple

def load_data(args, logger):
    """
    @params args: config args
    @params logger: logger
    @return images: [n_imgs, W, H, 3], ndarray
    @return poses: [n_imgs, 4, 4] (blender) or [n_imgs, 3, 4] (llff) camera poses from dataset, ndarray
    @return render_poses: [n_camera, 4, 4] (blender) or [n_imgs, 3, 5] (llff) render poses, tensor
    @return hwf: [height, width, focal] ndarray
    @return i_split: (i_train, i_val, i_test) tuple
    @return near: float
    @return far: float
    """
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = loader.load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        logger.info('Loaded llff %s %s %s %s', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            logger.info('Auto LLFF holdout %s,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        i_split = (i_train, i_val, i_test)

        logger.info('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        logger.info('NEAR FAR %s %s', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = loader.load_blender_data(args.datadir, args.half_res, args.testskip)
        logger.info('Loaded blender %s %s %s %s', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        
        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = loader.load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.
    else:
        logger.critical('Unknown dataset type %s exiting', args.dataset_type)
        exit(-1)

    return images, poses, render_poses, hwf, i_split, near, far
