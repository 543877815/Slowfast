import sys
import onnxruntime as ort
import torchvision.transforms as transforms
sys.path.insert(0, '../')
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import os
import numpy as np
import torch
import time
import slowfast.utils.logging as logging
from slowfast.datasets import utils as utils
from mytools.jinyang.utils import *
import PIL.Image as Image
from tqdm import tqdm
logger = logging.get_logger(__name__)
import torchvision

def mytest_onnx(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    ort_session = ort.InferenceSession("../onnxes/slowfast_rgb2.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    frame_path = r"D:\jupyter\SlowFast\test_imgs\image\3"
    frames = os.listdir(frame_path)
    test_times = 1
    starttime = time.time()

    for path in frames:
        img_path = os.path.join(frame_path, path)
        path_to_videos = [os.path.join(img_path, img) for img in os.listdir(img_path)]
        inputs = get_frames(cfg=cfg, path_to_videos=path_to_videos)

        for i in range(len(inputs)):
            inputs[i] = np.array(inputs[i].unsqueeze(0)).astype(np.float32)

        outputs = ort_session.run(
            None,
            {
                "x0": inputs[0],
                "x1": inputs[1],
                "bboxes": None
            },
        )
        a = torch.tensor(inputs[1]).squeeze(0).transpose(0, 1)
        dataframe = time.time()
        classes = np.argmax(outputs[0], 1)  # classes.shape = torch.Size([8])
        torchvision.utils.save_image(a, f'./debug/{dataframe}-{classes.item()}.png')
        print("predicts: {}".format(classes))

    endtime = time.time()
    second = (endtime - starttime)
    print("total time: {}, average time: {}".format(second, second / (test_times * len(frames))))


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=mytest_onnx)
