import jinyang
from jinyang.camera import ModelProcess, CameraProcess
from slowfast.utils.parser import load_config, parse_args
from multiprocessing import Pipe
import os
import numpy as np
import time


def mytest_trt(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    p1, p2 = Pipe(duplex=True)
    p3, p4 = Pipe(duplex=False)
    camera = CameraProcess(xname='camera',
                           pipe=p1)
    model = ModelProcess(xname='model',
                         pipe=p2,
                         main_pipe=p4,
                         slowfast_path="../engines/slowfast_rgb.plan",
                         yolov5_path="../checkpoints/yolov5m_4.pt",
                         cfg=cfg,
                         debug=False,
                         show_video=True)

    camera.start()
    model.start()

    while True:
        data = p3.recv()
        print(data['pred_of_yolov5'])
        print(data['prob_of_yolov5'])
        print(data['pred_of_slowfast'])
        print(data['prob_of_slowfast'])


if __name__ == "__main__":
    print(jinyang, jinyang.__version__)
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    mytest_trt(cfg=cfg)
