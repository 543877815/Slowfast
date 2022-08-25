import cv2 as cv
from slowfast.utils.parser import load_config, parse_args
import time
from multiprocessing import Process, Pipe, Lock
import numpy as np
from utils import get_frames, do_inference, load_engine, allocate_buffers
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.general import (check_img_size, check_imshow, Profile, increment_path, non_max_suppression, scale_coords)
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.plots import Annotator, colors, save_one_box
import tensorrt as trt
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class CameraProcess(Process):
    def __init__(self, xname, pipe, save_image=False,  source=0):
        super(CameraProcess, self).__init__()
        self.xname = xname
        self.pipe = pipe
        self.save_image = save_image
        self.source = source
        if self.save_image:
            os.makedirs('./image', exist_ok=True)

        self.width = 441
        self.height = 331
        self.maxsize = 64

    def run(self):
        # 打开摄像头
        self.cap = cv.VideoCapture(self.source)
        print("camera ready!")

        # print(self.cap.set(cv.CAP_PROP_AUTO_WB, 0))  # 关闭白平衡，解决偏色问题
        # print(self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)) #设置曝光为手动模式
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度

        id = 0
        while True:
            # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
            hx, frame = self.cap.read()
            frame = cv.cvtColor(frame, cv.IMREAD_COLOR)

            # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
            if hx is False:
                # 打印报错
                print('read video error')
                # 退出程序
                exit(0)

            # 显示摄像头图像，其中的video为窗口名称，frame为图像

            # frame = cv.resize(frame, (self.width, self.height), cv.INTER_NEAREST)
            # if self.show_video:
            #     cv.imshow('video', frame)
            # if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            #     break

            if self.save_image:
                cv.imwrite('./images/{}.png'.format(str(id).zfill(5)), frame)
            id += 1

            timeframe = time.time()
            self.pipe.send([frame, timeframe])

            # 监测键盘输入是否为q，为q则退出程序
            fps = 30
            if cv.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # 按q退出
                break


class ModelProcess(Process):
    def __init__(self, xname, pipe, main_pipe,
                 slowfast_path="../engines/slowfast_grayscale.plan",
                 yolov5_path="../checkpoints/yolov5s.pt",
                 cfg=None, debug=True, show_video=True, verbose=False):
        super().__init__()
        self.xname = xname
        self.pipe = pipe
        self.main_pipe = main_pipe
        self.cfg = cfg
        self.slowfast_path = slowfast_path
        self.debug = debug
        self.verbose = verbose
        self.show_video = show_video

        if self.debug:
            os.makedirs("./debug", exist_ok=True)

        # for yolov5
        self.yolov5_path = yolov5_path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        self.nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        self.project = ROOT / 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        self.timeframes = []
        self.frames = []
        self.isStart = []
        self.result = []

    def run(self):
        # slowfast model
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        slowfast_model = load_engine(trt_runtime=trt_runtime, plan_path=self.slowfast_path)

        # yolov5
        device = select_device("")
        data = ROOT / 'config/data.yaml'
        yolov5_model = DetectMultiBackend(self.yolov5_path, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = yolov5_model.stride, yolov5_model.names, yolov5_model.pt
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        yolov5_model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        # Dataloader
        test_transform = transforms.Compose([
            # transforms.transforms.Resize([256, 256]),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        batch_size = 1
        maxsize = 75

        self.isStart = []
        self.timeframes = []
        self.frames = []
        self.probabilities = []
        while True:

            frame, timeframe = self.pipe.recv()
            annotator = Annotator(frame, line_width=self.line_thickness, example=str(names))

            im0 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Input
            with dt[0]:
                im = test_transform(Image.fromarray(im0)).to(device)
                im = im.half() if yolov5_model.fp16 else im.float()  # uint8 to fp16/32
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = yolov5_model(im, augment=False, visualize=False)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            bbx = []
            for i, det in enumerate(pred):  # per image
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    bbx = det
                for *xyxy, conf, cls in reversed(det):
                    if self.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            frame = annotator.result()

            if view_img:
                cv.imshow('video', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                    break

            self.frames.append(frame)
            self.timeframes.append(timeframe)
            self.isStart.append(len(bbx))
            self.probabilities.append(bbx[0][4].item() if len(bbx) else 0)
            if len(self.frames) >= maxsize and self.isStart[0] and self.probabilities[0] > 0.5:
                inputs = get_frames(self.cfg, self.frames)
                h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream = allocate_buffers(slowfast_model, batch_size, trt.float32)
                out = do_inference(slowfast_model, inputs, h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream)
                prob = self.softmax(out)
                pred = np.argmax(out)
                for i in range(maxsize):
                    self.isStart[i] = 0
                    self.probabilities[i] = 0
                if self.verbose:
                    print(pred, prob)
                if self.debug:
                    curr_time = time.time()
                    a = torch.tensor(inputs[1].clone().detach()).squeeze(0).transpose(0, 1)
                    torchvision.utils.save_image(a, f'./debug/{curr_time}-{pred}.png')

                self.main_pipe.send([pred, prob])

            while len(self.timeframes) >= maxsize:
                self.frames.pop(0)
                self.timeframes.pop(0)
                self.isStart.pop(0)
                self.probabilities.pop(0)
    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits)
        probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return probs


def mytest_trt(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    p1, p2 = Pipe(duplex=True)
    p3, p4 = Pipe(duplex=False)
    mutex = Lock()
    camera = CameraProcess('camera', p1)
    model = ModelProcess('model', p2, p4, cfg=cfg, debug=True, mutex=mutex)

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
