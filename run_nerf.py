import os
import numpy as np
import imageio
import torch
from tqdm import tqdm, trange
import utils
from model import NeRF

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
np.random.seed(0)


def train(args, logger):
    # Load data
    images, poses, render_poses, hwf, i_split, near, far = utils.load_data(args, logger)
    i_train, i_val, i_test = i_split

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args
    )
    global_step = start

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    ############################################################
    # In test circuit
    # (Short circuit if only rendering out from trained model)
    ############################################################
    if args.render_only:
        print("RENDER ONLY")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            rgbs, _ = utils.render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                        savedir=testsavedir, render_factor=args.render_factor)
            print("Done rendering", testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                             utils.to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print("get rays")
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = np.stack([utils.get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print("done, concats")
        # [N_imgs, ro+rd+rgb (3), H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N_imgs, H, W, ro+rd+rgb (3), 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # train images only, [N_train_imgs, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # [N_imgs*H*W, ro+rd+rgb (3), 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print("shuffle rays")
        np.random.shuffle(rays_rgb)

        print("done")
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = args.N_iter + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    start = start + 1
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch : i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = utils.get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        ),
                        -1,
                    )
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),-1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = utils.render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                              verbose=i < 10, retraw=True,
                                              **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = utils.img2mse(rgb, target_s)
        trans = extras["raw"][..., -1]
        loss = img_loss
        psnr = utils.mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = utils.img2mse(extras["rgb0"], target_s)
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
            param_group["lr"] = new_lrate

        ############################################################
        # log and save
        ############################################################
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "network_fine_state_dict": render_kwargs_train[
                        "network_fine"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path
            )
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = utils.render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, "{}_spiral_{:06d}_".format(expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4",
                utils.to8b(disps / np.max(disps)),
                fps=30,
                quality=8,
            )

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                utils.render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[i_test],
                    savedir=testsavedir,
                )
            print("Saved test set")

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """
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
        embeddirs_fn, input_ch_views = utils.get_embedder(args.multires_views, args.i_embed)

    ################################
    # init nerf model
    ################################
    output_ch = 4  # density, color
    skips = [4]

    # first coarse MLP with stratified sampling only
    model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips, 
                 input_ch_views=input_ch_views,use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    # second fine MLP combined with importance sampling
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
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
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    #################################################
    # define render args dictionary in train and test
    #################################################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def run_network(pts, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
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
    outputs = torch.reshape(
        outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]]
    )
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


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # set up config parser
    parser = utils.config_parser()
    args = parser.parse_args()

    # set up logger
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, "audit.log")
    logger = utils.setup_logger(f)

    train(args, logger)
