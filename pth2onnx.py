from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import os
import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.models import build_model
from slowfast.datasets import utils as utils
import random

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    model.eval()

    # load pretrained network
    cu.load_test_checkpoint(cfg, model)

    ############ test for random input #######
    # batch_size = 8
    # img_channel = 3
    # fast_frame = 16
    # slow_frame = 64
    # img_width = 256
    # img_height = 256
    # inputs = [torch.zeros(batch_size, img_channel, fast_frame, img_width, img_height),
    # torch.zeros(batch_size, img_channel, slow_frame, img_width, img_height)]
    ##########################################


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=test)
