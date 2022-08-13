import sys

sys.path.insert(0, '../')
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
import time
logger = logging.get_logger(__name__)
from tqdm import tqdm

# 相同的间隔取帧序列
def get_seq_frames(cfg, path_to_videos):
    """
    Given the video index, return the list of sampled frame indexes.
    Args:
        index (int): the video index.
    Returns:
        seq (list): the indexes of frames of sampled from the video.
    """
    num_frames = cfg.DATA.NUM_FRAMES
    video_length = len(path_to_videos)

    seg_size = float(video_length - 1) / num_frames
    seq = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


# 根据帧序列返回16和64张图像帧
def get_frames(cfg, path_to_videos):
    path_to_videos.sort()
    seq = get_seq_frames(cfg, path_to_videos)

    frames = torch.as_tensor(
        utils.retry_load_images(
            [path_to_videos[frame] for frame in seq],
        )
    )

    # Perform color normalization.
    DATA_MEAN = [0.45, 0.45, 0.45]
    DATA_STD = [0.225, 0.225, 0.225]
    frames = utils.tensor_normalize(
        frames, DATA_MEAN, DATA_STD
    )

    min_scale, max_scale, crop_size = [cfg.DATA.TEST_CROP_SIZE] * 3
    NUM_SPATIAL_CROPS = cfg.TEST.NUM_SPATIAL_CROPS  # 1
    RANDOM_FLIP = cfg.DATA.RANDOM_FLIP  # False
    INV_UNIFORM_SAMPLE = cfg.DATA.INV_UNIFORM_SAMPLE  # True
    spatial_temporal_idx = 0
    spatial_sample_index = (
            spatial_temporal_idx % NUM_SPATIAL_CROPS
    )

    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    # Perform data augmentation.
    frames = utils.spatial_sampling(
        frames,
        spatial_idx=spatial_sample_index,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=RANDOM_FLIP,
        inverse_uniform_sampling=INV_UNIFORM_SAMPLE,
    )
    frames = utils.pack_pathway_output(cfg, frames)
    return frames


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

    test_times = 10
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
                preds = model(inputs[0], inputs[1])  # preds.shape = torch.Size([8, 174]) # TODO: reduce output shape to torch.Size([8, 13])
            classes = torch.argmax(preds, 1).cpu()  # classes.shape = torch.Size([8])
            # print("predicts: {}".format(classes))

    endtime = time.time()
    second = (endtime - starttime)
    print("total time: {}, average time: {}".format(second, second / (test_times * len(frames))))


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=mytest)
