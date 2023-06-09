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
    logger.info(parser.format_values())

    # load data
    images, poses, render_poses, hwf, i_split, near, far = utils.load_data(args, logger)
    i_train, i_val, i_test = i_split
    
    train(args)
