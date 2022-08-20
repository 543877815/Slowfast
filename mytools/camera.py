import cv2 as cv
import sys
import onnxruntime as ort
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.misc import launch_job
from multiprocessing import Process, Pipe
import datetime
from slowfast.datasets import utils as utils
import torch
import numpy as np
import time
import torchvision
import random
import torchvision.transforms as transforms
from PIL import Image
from mytools.utils import get_frames, do_inference, load_engine, allocate_buffers, allocate_buffers1, do_inference1
import tensorrt as trt


class CameraProcess(Process):
    def __init__(self, xname, pipe):
        super(CameraProcess, self).__init__()
        self.xname = xname
        self.pipe = pipe

        self.width = 256
        self.height = 256
        self.maxsize = 64

    def run(self):
        # 打开摄像头
        self.cap = cv.VideoCapture(0)
        print("camera ready!")

        self.cap.set(cv.CAP_PROP_AUTO_WB, 0) # 白平衡
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度

        id = 0
        while True:
            # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
            hx, frame = self.cap.read()
            # frame = cv.cvtColor(frame, cv.IMREAD_COLOR)

            # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
            if hx is False:
                # 打印报错
                print('read video error')
                # 退出程序
                exit(0)

            # 显示摄像头图像，其中的video为窗口名称，frame为图像

            frame = cv.resize(frame, (self.width, self.height), cv.INTER_NEAREST)
            cv.imshow('video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break

            timeframe = time.time()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.pipe.send([frame, timeframe])
            # cv.imwrite('./images/{}.png'.format(str(id).zfill(5)), frame)
            id += 1
            # self.q.append(frame)

            # 监测键盘输入是否为q，为q则退出程序
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break


class ModelProcess(Process):
    def __init__(self, xname, pipe,
                 slowfast_path="../engines/slowfast_rgb1.plan",
                 mobilenetv2_path="../engines/mobileNetv2.plan",
                 priority_queue=None, cfg=None, debug=False):
        super().__init__()
        self.xname = xname
        self.pipe = pipe
        self.priority_queue = priority_queue
        self.cfg = cfg
        self.slowfast_path = slowfast_path
        self.mobilenetv2_path = mobilenetv2_path
        self.mode = 'test'
        self.debug = debug
        self.timeframe = None
        self.q = None

        print("model ready!")
        self.model_ready = True

    def run(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        slowfast_model = load_engine(trt_runtime=trt_runtime, plan_path=self.slowfast_path)
        mobilenetv2_model = load_engine(trt_runtime=trt_runtime, plan_path=self.mobilenetv2_path)
        batch_size = 1
        maxsize = 100

        self.timeframe = []
        self.q = []

        while True:
            while len(self.timeframe) > 0 and time.time() - self.timeframe[0] > 3:
                self.timeframe.pop(0)
                self.q.pop(0)

            while len(self.timeframe) < maxsize:
                q, timeframe = self.pipe.recv()
                self.timeframe.append(timeframe)
                self.q.append(q)

            h_input1, d_input1, h_output, d_output, stream = allocate_buffers1(mobilenetv2_model, batch_size, trt.float32)
            out = do_inference1(mobilenetv2_model, self.q[0], h_input1, d_input1, h_output, d_output, stream)
            isStart = np.argmax(out)  # 1: active

            if isStart:
                inputs = get_frames(self.cfg, self.q)
                h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream = allocate_buffers(slowfast_model, batch_size, trt.float32)
                out = do_inference(slowfast_model, inputs, h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream)
                dataframe = time.time()
                classes = np.argmax(out)
                print("{}, predicts: {}".format(dataframe, classes))
                if self.debug:
                    a = torch.tensor(inputs[1]).squeeze(0).transpose(0, 1)
                    torchvision.utils.save_image(a, f'./debug/{dataframe}-{classes}.png')


def mytest_trt(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    p1, p2 = Pipe(duplex=True)
    camera = CameraProcess('camera', p1)
    model = ModelProcess('model', p2, cfg=cfg)

    camera.start()
    model.start()


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=mytest_trt)
