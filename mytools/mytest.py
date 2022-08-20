import sys

sys.path.insert(0, '../')
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import os
import numpy as np
import torch
import PIL.Image as Image
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.models import build_model
from slowfast.datasets import utils as utils
import random
import time
logger = logging.get_logger(__name__)
from tqdm import tqdm
from mytools.utils import get_frames


def mytest(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # load pretrained network
    cu.load_test_checkpoint(cfg, model)
    model.eval()

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
    frame_path = "/data/lifengjun/gesture_dataset/dataset/image/RGB/val/6/"
    frames = os.listdir(frame_path)

    test_times = 1
    starttime = time.time()

    for i in tqdm(range(test_times)):
        for path in frames:
            img_path = os.path.join(frame_path, path)
            path_to_videos = [os.path.join(img_path, img) for img in os.listdir(img_path)]
            inputs = get_frames(cfg=cfg, path_to_videos=path_to_videos)

            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].unsqueeze(0)  # 增加batch的维度
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Perform the forward pass.
            with torch.no_grad():
                preds = model(inputs)  # preds.shape = torch.Size([8, 13])
            classes = torch.argmax(preds, 1).cpu()  # classes.shape = torch.Size([8])
            print("predicts: {}".format(classes))

    endtime = time.time()
    second = (endtime - starttime)
    print("total time: {}, average time: {}".format(second, second / (test_times * len(frames))))


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=mytest)
