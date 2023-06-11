import torch
import numpy as np
import time
import os
import utils
import imageio
import torch.nn.functional as F


def get_rays(H, W, focal, c2w):
    """

    generate ray (origin, direction) for all pixels
    @return rays_d: [W, H, 3]
    @return rays_o: [W, H, 3]
    """
    # meshgrid return two W * H matrix, each element in i and j at the same position represent this pixel's coordinate
    # pytorch's meshgrid has indexing='ij'

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    """
    1. convert screen space coordinate to camera coordinate first
    assuming (i, j) is screen coordinate and (x, y, z) is camera coordinate 
    according to pinhole, i / f = x / z and j / f = y / z (z=1 at default)
    move the i j coordinate by (-W/2, -H/2) offset because we want the camera facing to the center of screen
    also the y coordinate in screen and camera is up-side down (remember pinhole)
    reference: https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    dir = (i - 0.5W / focal, -(j - 0.5H) / focal, -1) (screen is at z=-1)
    2. convert from camera coordinate to world by c2w dot dir
    """
    dirs = torch.stack(
        [(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -torch.ones_like(i)], -1
    )  # stack dir vectors in col
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """
    get_rays numpy version
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))

    dirs = np.stack(
        [(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -np.ones_like(i)], -1
    )
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def render_path(
    render_poses, hwf, chunk, render_kwargs, savedir=None, render_factor=0
):
    """
    render points along the path (for all render poses)
    @parameter render_poses: camera pose in rendering (position & view directions)
    @parameter hwf: camera intrinstic params
    @parameter chunk: chunk size
    @parameter render_kwargs: render args (including the model network, far, near, etc..)
    @parameter savedir: diretory to save result image
    @parameter render_factor: factor
    @return rgbs: ???
    @return disps: disparity (inverse depth)
    """
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render(
    H,
    W,
    focal,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs
):
    """Render rays for single camera
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # generate rays to render fill image (when called in render_path)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide normalized ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # flatten viewdirs to shape [batch_size, 3] (batch_size = W * H)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

    sh = rays_d.shape  # [batch_size, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

    # Create ray batch, flatten rays_o, rays_d to [batch_size, 3]
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # (ray origin, ray direction, min dist, max dist) for each ray
    # for example:
    # rays_d = torch.Tensor([[1, 2, 3],
    #                        [4, 5, 6]]),
    # rays_o = torch.Tensor([[0.1, 0.2, 0.3],
    #                        [0.4, 0.5, 0.6]]),
    # near = torch.Tensor([[near,],
    #                      [near,]]),
    # far = torch.Tensor([[far,],
    #                      [far,]]),
    # ray = torch.Tensor([[0.1, 0.2, 0.3, 1, 2, 3, near, far],
    #                     [0.4, 0.5, 0.6, 4, 5, 6, near, far]]),
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.concat([rays_o, rays_d, near, far], axis=-1)

    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = torch.concat([rays, viewdirs], axis=-1)

    # render rays in batch, stack up return data to a map
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(rays[i: i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # v-stack map
    all_ret = {k: torch.concat(all_ret[k], 0) for k in all_ret}

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])  # (batch_size, ...)
        # reshape
        # i.e. rgb_map reshape from [N_rays, 3] to [batch_size, 3] (the same...each pixel only shoot a ray)
        # ? feel like this is the same shape...
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # a list of map consist of rgb_map, disp_map, acc_map
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    # other data of the dictionary
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_rays(
    rays,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False
):
    """Volumetric rendering along a ray, in batch

    Args:
      ray: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    ###########################
    # generate points along ray
    ###########################

    # extract near and far from rays
    # bounds tensor([[[n., f.]],
    #                [[n., f.]],
    #                 ...
    #                [[n., f.]]])
    # -1: add addition dimenstion
    bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])  # [-1, batch_size, 2]
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]?? [batch_size, 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)

    if not lindisp:
        # z = near * (1 - t) + far * t
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        # interpolate with inverse depth
        # 1/z = 1/near * (1 - t) + 1/far * t
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    N_rays = rays.shape[0]  # batch size
    # shape [N_rays, N_samples], copy z across samples
    z_vals = z_vals.expand([N_rays, N_samples])

    # pertub sample
    if perturb > 0:
        # get the middle points of each intervals
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # get the intervals between middle points (and first & last point)
        upper = torch.concat([mids, z_vals[..., -1:]], -1)
        lower = torch.concat([z_vals[..., :1], mids], -1)
        # random perturb
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # [N_rays, 3]
    if rays.shape[-1] > 8:
        viewdirs = rays[:, -3:]
    else:
        viewdirs = None

    # Points in space to evaluate model at.
    # pts = o + t * d
    # [N_rays, 1, 3] + [N_rays, 1, 3] * [N_rays, n_samples, 1] = [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + z_vals[..., :, None] * rays_d[..., None, :]

    ###################
    # coarse network
    ###################

    # apply the network to the points
    raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd)

    ###################
    # finer network
    ###################
    if N_importance > 0:
        rgb_map_coarse, disp_map_coarse, acc_map_coarse = rgb_map, disp_map, acc_map

        # importance sampling
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        # generate points
        z_vals, _ = torch.sort(torch.concat([z_vals, z_samples], -1), -1)
        # [N_rays, 1, 3] + [N_rays, N_samples + N_importance, 1] * [N_rays, 1, 3] = [N_rays, N_samples + N_importance, 3]
        pts = rays_o[..., None, :] + z_vals[..., :, None] * rays_d[..., None, :]

        # run network_fine again
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd)

    # return a length-varied map
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_coarse
        ret['disp0'] = disp_map_coarse
        ret['acc0'] = acc_map_coarse
        ret['z_std'] = torch.std(z_samples, -1)  # [N_rays]

    if verbose:
        for k in ret:
            if (torch.isnan(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan.")
            elif (torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains inf.")

    return ret


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False, white_bkgd=False):
    """
    convert model's output to rgb, disparity, accuracy, weights, and depth

    @parameter raw: [N_rays, N_samples, 4]
    @parameter z_vals: [N_rays, N_samples]
    @parameter rays_d: [N_rays, 3]
    @parameter raw_noise_std: noise std
    @parameter pytest: in test circuit
    @parameter white_bkgd: white background
    Returns:
      rgb_map: [N_rays, 3]. Estimated RGB color of a ray.
      disp_map: [N_rays]. Disparity map. Inverse of depth map.
      acc_map: [N_rays]. Sum of weights along each ray.
      weights: [N_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [N_rays]. Estimated distance to object.
    """

    ########
    #  rgb
    ########
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    ##########
    # distance
    ##########
    # distance between t_(i+1) - t_i, delta in the paper
    # the last distance is infinity
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.concat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    

    ##########
    # density
    ##########
    noise = 0
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # density = 1 - exp(- sigma * delta)
    # higher density imply higher likelihood of being absorbed at this point.
    # add noise to regularize network during training and prevent floater artifacts
    # [N_rays, N_samples]
    density = 1. - torch.exp(-F.relu(raw[..., 3] + noise) * dists)

    ##########
    # weight
    ##########

    # weight = transmittance * density
    # transimttance = \prod_{j=1}^{i} (1 - density_1)
    # cumprod: cumulative product, i.e. cumprod([2, 3, 4]) = [2, 6, 24]
    # exclusive cumprod is needed here i.e. cumprod([2, 3, 4]) = [1, 2, 6]
    # add dumpy 1 in the head of list, and remove the last one, i.e. [1, 2, 3, 4] = [1, 2, 6]
    weights = density * torch.cumprod(torch.cat([torch.ones(
        (density.shape[0], 1)), 1.-density + 1e-10], -1), -1)[:, :-1]  # [N_rays, N_samples]

    # compute weighted color
    # C(r) = sum_{i}^{N_sample} weight * color
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    # compute weighted depth (estimated depth)
    depth_map = torch.sum(weights * z_vals, -1)  # [N_rays]

    # compute weighted inverse depth (disparity)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))  # [N_rays]

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, -1)  # [N_rays]

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def sample(z_vals, weights, N_importance, det=False, pytest=False):
    """
    hierarchical sampling
    sample points according to weight (pdf)
    @parameter z_vals: [N_rays, N_samples-1], bin, discrete random variable
    @parameter weights:  [N_rays, N_samples-1]
    @parameter N_importance: bool
    @parameter det: bool perturb == 0
    @return: [N_rays, N_importance, 1], points
    """
    # get pdf and cdf
    weights = weights + 1e-5 #! do not use += to avoid inplace operation
    # w_i = w_i/\sum_{j=0}^{N} w_j
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)  # cdf = [w0, w0+w1, w0+w1+w2, ....]
    # (N_rays, len(bins))
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        # linear sample
        u = torch.linspace(0., 1., steps=N_importance)
        # (N_rays, len(bins), N_importance)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        # uniform random samples
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance])

    if pytest:
       # fix random seed
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_importance]
        if det:
            u = np.linspace(0., 1., N_importance)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # invert CDF
    # reference: https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html
    # find the index of variable k which satisfy sum_{i}^{k-1} p_i < u <= sum_{i}^{k} p_i
    u = u.contiguous()
    idx = torch.searchsorted(cdf, u, right=True)
    # lower bound idx (at least 0)
    lower = torch.max(torch.zeros_like(idx-1), idx-1)
    # uppber bound idx (at most N_importance_sample)
    upper = torch.min(idx, (cdf.shape[-1]-1) * torch.ones_like(idx))
    idx_group = torch.stack([lower, upper], -1)  # (N_rays, N_importance, 2)

    # gather slices from params axis axis according to indices
    matched_shape = [idx_group.shape[0], idx_group.shape[1],
                     cdf.shape[-1]]  # (N_rays, N_importance, len(bins))
    # unsqueeze: cdf first reshape to (N_rays, 1, len(bins)) and then reshape to (N_rays, N_importance, len(bins))
    # gather: cdf_group[i][j][k] = cdf[i][j](idx_group[i][j][k]), i range (0, N_rays), j range (0, N_importance_sample), k range (0, 2)
    cdf_group = torch.gather(cdf.unsqueeze(1).expand(
        matched_shape), 2, idx_group)  # (N_rays, N_importance, 2)
    z_group = torch.gather(z_vals.unsqueeze(1).expand(
        matched_shape), 2, idx_group)  # (N_rays, N_importance, 2)

    # length of selected cdf interval
    denom = (cdf_group[..., 1] - cdf_group[..., 0])
    # set to 1 if interval length is too small
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_group[..., 0]) / denom
    # [N_rays, N_importance, 1]
    samples = z_group[..., 0] + t * (z_group[..., 1] - z_group[..., 0])

    return samples


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    # allow us to linear mapping t [0, 1] to origin space z [n to inf] in disparity
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    # todo: why not multiply with matrix directly...
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d