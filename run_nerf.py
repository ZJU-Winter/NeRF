import utils
import torch
import numpy as np
import os
import model.NeRF as NeRF
import imageio
import time
from tqdm import tqdm, trange

torch.autograd.set_detect_anomaly(True)
logger = utils.logger


def train(args):
    ################################################
    # load data
    ################################################
    images, poses, render_poses, hwf, i_split, near, far = utils.load_data(
        args)
    i_train, i_val, i_test = i_split

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # cast intrinsics to right types (int)
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    logger.info("images {}".format(images.shape))
    logger.info("poses {}".format(poses.shape))
    logger.info("render_poses {}".format(render_poses.shape))
    # print("hwf", hwf)
    # print("i_split", i_split)
    # print("near/far", near.shape, far.shape)

    # log configured args
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        logger.info('{} = {}'.format(arg, attr))

    ################################################
    # create nerf model
    ################################################
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args)
    global_step = start

    # add near and far key-value to render args dictionary
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU

    render_poses = torch.Tensor(render_poses).to(utils.device)

    ############################################################
    # In test circuit
    # (Short circuit if only rendering out from trained model)
    ############################################################
    basedir = args.basedir
    expname = args.expname
    if args.render_only:
        logger.info('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape {render_poses.shape}')

            rgbs, _ = utils.render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                        savedir=testsavedir, render_factor=args.render_factor)
            logger.info('Done rendering'.format(testsavedir))
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                             utils.to8b(rgbs), fps=30, quality=8)

            return

    ############################################################
    # optimize loop
    ############################################################
    N_rand = args.N_rand

    # during each iteration the entire batch of rays are only sampled from a single image.
    # When it is false, we sample the rays from all of the images during each iteration.
    # setting it true when using synthetic images is recommended
    # reference: https://github.com/bmild/nerf/issues/108
    # use_batching = not args.no_batching
    use_batching = True

    if use_batching:
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [utils.get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N_camera, ro+rd (2), H, W, 3]
        # [N_imgs, ro+rd+rgb (3), H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N_imgs, H, W, ro+rd+rgb (3), 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # train images only, [N_train_imgs, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)
        # [N_imgs*H*W, ro+rd+rgb (3), 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        logger.info('shuffle rays')
        np.random.shuffle(rays_rgb)
        logger.info('done')
        i_batch = 0

        # move to gpu
        images = torch.Tensor(images).to(utils.device)
        rays_rgb = torch.Tensor(rays_rgb).to(utils.device)

    poses = torch.Tensor(poses).to(utils.device)

    logger.info('Begin')
    logger.info('TRAIN views are {}'.format(i_train))
    logger.info('TEST views are {}'.format(i_test))
    logger.info('VAL views are {}'.format(i_val))
    N_iters = args.N_iter

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        if use_batching:
            # select a random batch size
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, (ro+rd+rgb)3, 3]
            batch = torch.transpose(batch, 0, 1)  # [ro+rd+rgb(3), B, 3]
            # extract rays and imgs from batch
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        # todo
        # for no_batching, sample more rays from one image

        rgb, disp, acc, extras = utils.render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                              verbose=i < 10, retraw=True,
                                              **render_kwargs_train)
        
        optimizer.zero_grad()
        img_loss = utils.img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = utils.mse2psnr(img_loss)

        # Add MSE loss for coarse-grained model
        if 'rgb0' in extras:
            img_loss0 = utils.img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = utils.mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time()-time0

        ############################################################
        # log and save
        ############################################################

        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            logger.info('Saved checkpoints at {}'.format(path))

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = utils.render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
            logger.info(f'Done, saving {rgbs.shape} {disps.shape}')
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             utils.to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info('test poses shape {}'.format(poses[i_test].shape))
            with torch.no_grad():
                utils.render_path(torch.Tensor(poses[i_test]).to(
                    utils.device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            logger.info('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            if i % args.i_img == 0 and i > 0:
                # log a validation view
                # img_i = np.random.choice(i_val)
                img_i = 3
                target = images[img_i]
                pose = poses[img_i, :3, :4]
                with torch.no_grad():
                    rgb, disp, acc, extras = utils.render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test)
                psnr = utils.mse2psnr(utils.img2mse(rgb, target))
                # Save out the validation image
                valimgdir = os.path.join(basedir, expname, 'val_imgs')
                os.makedirs(valimgdir, exist_ok=True)
                if i == args.i_img:
                    imageio.imwrite(os.path.join(valimgdir, 'target.png'), utils.to8b(target.cpu().numpy()))
                imageio.imwrite(os.path.join(valimgdir, '{:06d}.png'.format(i)), utils.to8b(rgb.cpu().numpy()))
                logger.info(f'Saved {i} validation images. psnr: {psnr.item()}')


        global_step += 1


def create_nerf(args):
    """
    instantiate nerf MLP model
    """

    ################################
    # positionol embedding
    ################################

    # embed functions for positioin
    embed_fn, input_ch = utils.get_embedder(args.multires, args.i_embed)

    # embed functions for view direction
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = utils.get_embedder(
            args.multires_views, args.i_embed)

    ################################
    # init nerf model
    ################################
    output_ch = 4  # density, color
    skips = [4]

    # first coarse MLP with stratified sampling only
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(utils.device)
    grad_vars = list(model.parameters())

    # second fine MLP combined with importance sampling
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(utils.device)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    ###############################################
    # create optimizer and Load checkpoints
    ###############################################

    # optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info('Found ckpts {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info('Reloading from {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    #################################################
    # define render args dictionary in train and test
    #################################################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        logger.info('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def run_network(pts, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    encode input points and applies network 'fn'.
    @paramter pts: points' position 
    @paramter viewdirs: points' view direction
    @fn: network function (i.e. model and model_fine)
    @embed_fn: encoding functions
    @embeddirs_fn: encoding functions for viewdirs
    @netchunk: chunk size 
    @return: output after encoding and applied network fn (the shape should be (W, H, N_samples, 4)?)
    """

    # flatten pts, i.e. if pts shape is (W, H, N_sample, P), flatten to (W * H * N_sample, P), where P is 3 for postion
    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    embedded = embed_fn(pts_flat)

    # add viewdirs into points beside position
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    # reshape back to origin shape (W, H, N_samples, O) O is the output dimension (which should be 4)
    outputs = torch.reshape(outputs_flat, list(
        pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


if __name__ == '__main__':
    # set up config parser
    parser = utils.config_parser()
    args = parser.parse_args()

    # set up logger
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, "audit.log")
    utils.setup_logger(f)
    train(args)
