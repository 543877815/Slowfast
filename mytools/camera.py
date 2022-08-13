import cv2 as cv
import sys
import onnxruntime as ort
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.misc import launch_job
from multiprocessing import Process, Pipe, Manager
import datetime
from slowfast.datasets import utils as utils
import torch
import numpy as np
import time


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

        # cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度

        last_time = time.time()

        while True:
            # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
            hx, frame = self.cap.read()
            # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
            if hx is False:
                # 打印报错
                print('read video error')
                # 退出程序
                exit(0)

            # if len(self.q) > self.maxsize:
            #     self.q.pop(0)
            #     interval = time.time() - last_time
            # if self.pipe.poll():
            #     self.pipe.send(self.q)

            # 显示摄像头图像，其中的video为窗口名称，frame为图像
            # cv.imshow('video', frame)

            frame = cv.resize(frame, (self.width, self.height), cv.INTER_NEAREST)

            self.pipe.send([frame, time.time()])
            # self.q.append(frame)

            # 监测键盘输入是否为q，为q则退出程序
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break


class ModelProcess(Process):
    def __init__(self, xname, pipe, priority_queue=None, cfg=None):
        super().__init__()

        self.xname = xname
        self.pipe = pipe
        self.priority_queue = priority_queue
        self.cfg = cfg

        print("model ready!")
        self.model_ready = True

    # 相同的间隔取帧序列
    def get_seq_frames(self, path_to_videos):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(path_to_videos)

        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2)

        return seq

    def get_frames(self, q):
        seq = self.get_seq_frames(q)

        frames = torch.as_tensor(
            np.stack([q[frame] for frame in seq])
        )

        # Perform color normalization.
        DATA_MEAN = [0.45, 0.45, 0.45]
        DATA_STD = [0.225, 0.225, 0.225]
        frames = utils.tensor_normalize(
            frames, DATA_MEAN, DATA_STD
        )

        min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        NUM_SPATIAL_CROPS = self.cfg.TEST.NUM_SPATIAL_CROPS  # 1
        RANDOM_FLIP = self.cfg.DATA.RANDOM_FLIP  # False
        INV_UNIFORM_SAMPLE = self.cfg.DATA.INV_UNIFORM_SAMPLE  # True
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
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames

    def run(self):
        model_path = ("../onnxes/slowfast_rgb.onnx")
        ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        maxsize = 64
        self.timeframe = []
        self.q = []

        while True:
            while len(self.timeframe) > 0 and time.time() - self.timeframe[0] > 2:
                self.timeframe.pop(0)
                self.q.pop(0)

            while len(self.timeframe) < maxsize:
                q, timeframe = self.pipe.recv()
                self.timeframe.append(timeframe)
                self.q.append(q)
                cv.imshow('video', self.q[-1])
                if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                    break

            inputs = self.get_frames(self.q)

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
            classes = np.argmax(outputs[0], 1)  # classes.shape = torch.Size([8])
            print("{}, predicts: {}".format(time.time(), classes))


def mytest_onnx(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    p1, p2 = Pipe(duplex=True)
    camera = CameraProcess('camera,', p1)
    model = ModelProcess('model', p2, cfg=cfg)

    camera.start()
    model.start()


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=mytest_onnx)
