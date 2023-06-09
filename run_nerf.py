import utils
import torch
import numpy as np

def train(args):
    pass 

if __name__=='__main__':
    # set up config parser
    parser = utils.config_parser()
    args = parser.parse_args()

    # set up logger
    logger = utils.setup_logger(args.logpath)

    # load data
    images, poses, render_poses, hwf, i_split, near, far = utils.load_data(args, logger)
    i_train, i_val, i_test = i_split

    # cast intrinsics to right types (int)
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # print("images", images)
    # print("poses", poses)
    # print("render_poses", render_poses)
    # print("hwf", hwf)
    # print("i_split", i_split)
    # print("near/far", near, far)
    
    # log configured args
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        logger.info('{} = {}'.format(arg, attr))





    train(args)
