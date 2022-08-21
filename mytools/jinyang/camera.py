import cv2 as cv
from jinyang.slowfast.utils.parser import load_config, parse_args
import time
from multiprocessing import Process, Pipe
import numpy as np
from jinyang.utils import get_frames, do_inference, load_engine, allocate_buffers, allocate_buffers1, do_inference1
import tensorrt as trt
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from queue import Queue

class CameraProcess(Process):
    def __init__(self, xname, pipe, show_video=True, save_image=False):
        super(CameraProcess, self).__init__()
        self.xname = xname
        self.pipe = pipe
        self.show_video = show_video
        self.save_image = save_image
        if self.save_image:
            os.makedirs('./image', exist_ok=True)

        self.width = 441
        self.height = 331
        self.maxsize = 64

    def run(self):
        # 打开摄像头
        self.cap = cv.VideoCapture(0)
        print("camera ready!")

        # print(self.cap.set(cv.CAP_PROP_AUTO_WB, 0))  # 关闭白平衡，解决偏色问题
        # print(self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)) #设置曝光为手动模式
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

            # frame = cv.resize(frame, (self.width, self.height), cv.INTER_NEAREST)
            if self.show_video:
                cv.imshow('video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break

            if self.save_image:
                cv.imwrite('./images/{}.png'.format(str(id).zfill(5)), frame)
            id += 1

            timeframe = time.time()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.pipe.send([frame, timeframe])

            # 监测键盘输入是否为q，为q则退出程序
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break


class ModelProcess(Process):
    def __init__(self, xname, pipe, main_pipe,
                 slowfast_path="../engines/slowfast_grayscale.plan",
                 mobilenetv2_path="../engines/mobileNetv2.plan",
                 cfg=None, debug=False, verbose=False):
        super().__init__()
        self.xname = xname
        self.pipe = pipe
        self.main_pipe = main_pipe
        self.cfg = cfg
        self.slowfast_path = slowfast_path
        self.mobilenetv2_path = mobilenetv2_path
        self.debug = debug
        self.verbose = verbose

        if self.debug:
            os.makedirs("./debug", exist_ok=True)

        self.timeframes = []
        self.frames = []
        self.isStart = []
        self.result = []

    def run(self):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        slowfast_model = load_engine(trt_runtime=trt_runtime, plan_path=self.slowfast_path)
        mobilenetv2_model = load_engine(trt_runtime=trt_runtime, plan_path=self.mobilenetv2_path)
        print("model ready!")
        batch_size = 1
        maxsize = 100
        test_transform = transforms.Compose([
            transforms.transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.504, 0.511, 0.486],
                                 std=[0.300, 0.291, 0.286])
        ])
        self.timeframes = []
        self.frames = []
        self.isStart = []
        while True:
            while len(self.timeframes) > 0 and time.time() - self.timeframes[0] > 3:
                self.timeframes.pop(0)
                self.frames.pop(0)
                self.isStart.pop(0)

            while len(self.timeframes) < maxsize:
                q, timeframe = self.pipe.recv()
                self.timeframes.append(timeframe)
                self.frames.append(q)
                q = test_transform(Image.fromarray(q))
                h_input1, d_input1, h_output, d_output, stream = allocate_buffers1(mobilenetv2_model, batch_size, trt.float32)
                out = do_inference1(mobilenetv2_model, q, h_input1, d_input1, h_output, d_output, stream)
                self.isStart.append(np.argmax(out))  # 1: active

            # if self.isStart[0]:
            #     for i in range(64):
            #         self.isStart[i] = 0
            start_time = self.timeframes[0]
            end_time = self.timeframes[maxsize - 1]
            inputs = get_frames(self.cfg, self.frames)
            h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream = allocate_buffers(slowfast_model, batch_size, trt.float32)
            out = do_inference(slowfast_model, inputs, h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream)

            # result
            self.main_pipe.send(out)
            # if len(self.result) > maxsize:
            #     self.result.pop(0)
            # self.result.append(out)

            classes = np.argmax(out)
            self.timeframes = self.timeframes[:64]
            self.frames = self.frames[:64]
            self.isStart = self.isStart[:64]

            if self.verbose:
                print("current time: {}, time cost: {}, predicts: {}".format(time.time(), end_time - start_time, classes))
            if self.debug:
                a = torch.tensor(inputs[1]).squeeze(0).transpose(0, 1)
                torchvision.utils.save_image(a, f'./debug/{end_time}-{classes}.png')


def mytest_trt(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    p1, p2 = Pipe(duplex=True)
    p3, p4 = Pipe(duplex=False)
    camera = CameraProcess('camera', p1)
    model = ModelProcess('model', p2, p4, cfg=cfg)

    camera.start()
    model.start()

    while True:
        print(np.argmax(p3.recv()))


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)

    mytest_trt(cfg=cfg)
