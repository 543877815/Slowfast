import tensorrt as trt
import torch
import cv2 as cv
from slowfast.utils.parser import load_config, parse_args
import time
from multiprocessing import Process, Pipe
import numpy as np
from jinyang.utils import get_frames, do_inference, load_engine, allocate_buffers
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (check_img_size, check_imshow, Profile, increment_path, non_max_suppression, scale_coords)
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.plots import Annotator, colors, save_one_box
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
import sys
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class CameraProcess(Process):
    def __init__(self, xname, pipe, save_image=False, source=0):
        super(CameraProcess, self).__init__()
        self.xname = xname
        self.pipe = pipe
        self.save_image = save_image
        self.source = source
        if self.save_image:
            os.makedirs('./image', exist_ok=True)

        self.width = 640
        self.height = 640

    def run(self):
        # 打开摄像头
        self.cap = cv.VideoCapture(self.source)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)  # 设置宽度
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)  # 设置长度
        # self.cap.set(5,25)
        print("camera ready!")

        # print(self.cap.set(cv.CAP_PROP_AUTO_WB, 0))  # 关闭白平衡，解决偏色问题
        # print(self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)) #设置曝光为手动模式
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度
        id = 0
        while True:
            # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
            hx, frame = self.cap.read()

            # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
            if hx is False:
                # 打印报错
                print('read video error')
                # 退出程序
                exit(0)

            frame = cv.cvtColor(frame, cv.IMREAD_COLOR)
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
                 slowfast_path="../engines/slowfast_rgb.plan",  # 视频理解模型
                 yolov5_path="../checkpoints/yolov5m_6.pt",  # 手势检测模型
                 yolov5_face_path="../checkpoints/face_last.pt",  # 头手检测模型
                 face_data_yaml = r"mytools/jinyang/config/face_data.yaml",
                 gesture_data_yaml=r"mytools/jinyang/config/gesture_data.yaml",  # data.yaml的位置
                 cfg=None,  # slowfast 配置文件
                 debug=True,  # 是否进入debug模式
                 show_video=True,  # 是否展现摄像头
                 show_head=True,  # 是否展现头手
                 show_gesture=False,  # 是否展现手势
                 show_center=True,  # 是否展现中心点
                 verbose=False,  # 是否打印输出
                 decimal_precision=4,  # 小数精度
                 start_frame_clses=None,  # 作为slowfast的起始帧的识别的手势的类别
                 conf_yolov5=None  # yolov5置信度
                 ):
        super().__init__()
        self.xname = xname
        self.pipe = pipe
        self.main_pipe = main_pipe
        self.cfg = cfg
        self.slowfast_path = slowfast_path
        self.face_data_yaml = face_data_yaml
        self.gesture_data_yaml = gesture_data_yaml
        self.debug = debug
        self.verbose = verbose
        self.show_video = show_video
        self.show_gesture = show_gesture
        self.show_head = show_head
        self.decimal_precision = decimal_precision
        self.show_center = show_center

        if self.debug:
            os.makedirs("./debug", exist_ok=True)

        # for yolov5
        self.yolov5_path = yolov5_path
        self.yolov5_face_path = yolov5_face_path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.conf_yolov5 = conf_yolov5  # confidence of showing bounding box
        self.start_frame_clses = start_frame_clses

        if conf_yolov5 is None:
            self.conf_yolov5 = {"others": 0.5}
        else:
            assert isinstance(self.conf_yolov5, dict), f"The data type of conf_static is \"{type(self.conf_yolov5)}\", which should be a \"dict\" instead."
            self.conf_yolov5 = conf_yolov5
            if "others" not in self.conf_yolov5:
                self.conf_yolov5["others"] = 0.5
                logging.warning("\"others\" is not set in conf_static, which is set to 0.5 by default")

        self.start_frame_clses = start_frame_clses

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

    def inference(self, im0, yolov5_model, names, pred_conf, view_img, annotator, device):
        # Dataloader
        test_transform = transforms.Compose([
            # transforms.transforms.Resize([256, 256]),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        dt = (Profile(), Profile(), Profile())

        # Input
        with dt[0]:
            im = test_transform(Image.fromarray(im0)).to(device)
            im = im.half() if yolov5_model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with dt[1]:
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
                c = int(cls)  # integer class
                if conf < (pred_conf[names[c]] if names[c] in pred_conf else pred_conf["others"]):
                    continue
                if self.save_crop or view_img:  # Add bbox to image
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    if self.show_center:
                        center_x = (xyxy[0] + xyxy[2]) / 2
                        center_y = (xyxy[1] + xyxy[3]) / 2
                        center_xyxy = [center_x, center_y, center_x, center_y]
                        annotator.box_label(center_xyxy, label=None, color=colors(c, True))

        return bbx, annotator

    def getData(self, data, name):
        res = []
        cls = lambda x: x if x in self.conf_yolov5 else "others"  # 获取在conf_yolov5中的名称
        for i in range(len(data)):
            tmp = cls(name[int(data[i][-1])])
            if data[i][-2] > self.conf_yolov5[tmp]:
                data[i][-1] = name[int(data[i][-1])]
                data[i][-2] = round(data[i][-2], self.decimal_precision)
                res.append(data[i])
        return res

    def update_conf_yolov5(self, conf_yolov5):
        assert isinstance(conf_yolov5, dict), "conf_yolov5 should be a dict"
        assert "others" in conf_yolov5, "args [others] should be set in conf_yolov5"
        self.conf_yolov5 = conf_yolov5

    def judge_start(self,pred_of_yolov5,num = 8):                  #判断是否要进入slowfast进行预测，判断为start的标准是
        start = 0                                                                    #1、众数的个数不能大于30个(说明是手一直保持起始手势)
        mode_num = max([pred_of_yolov5.count(i) for i in set(list(pred_of_yolov5))]) #2、前8个预测里，有出现连续4个以上的同一个起始手势(说明不是偶然做出的起始手势,3除外)
        first_repeat_num = 1                                                               ###3、连续出现同一个手势的个数不能超过25个(2除外)
        repeat = pred_of_yolov5[0]
        all_repeat_num = 0
        start_frame_clses = self.start_frame_clses
        for i in range(1,num):
            if (pred_of_yolov5[i] in start_frame_clses) and pred_of_yolov5[i] == repeat:
                first_repeat_num +=1
            else:
                repeat = pred_of_yolov5[i]
                if first_repeat_num >= 3:
                    break
                first_repeat_num = 1


        repeat = pred_of_yolov5[0]

        for i in range(1, len(pred_of_yolov5)):
            if (pred_of_yolov5[i] in ['1','3','11']) and pred_of_yolov5[i] == repeat:
                all_repeat_num += 1
            else:
                repeat = pred_of_yolov5[i]
            if all_repeat_num >= 25:
                break

        if mode_num != 32 and all_repeat_num < 25:
            if pred_of_yolov5[0] == '3' or first_repeat_num >= 3 :
                start = 1

        # if start == 0:
        #     print(pred_of_yolov5)
        #     print(self.probabilities)
        #     print(mode_num,first_repeat_num,all_repeat_num,pred_of_yolov5[0])

        return start

    def filter_pred_prob(self,confidence=0.85):         ##过滤掉pred里面那些置信度小于阈值的预测

        for i in range(len(self.pre_preds)):
            if self.pre_preds[i] in ['2','3']:
                if self.probabilities[i] < 0.87:
                    self.probabilities[i] = 0
                    self.pre_preds[i] = '0'
            if self.pre_preds[i] in ['1','11','5']:
                if self.probabilities[i] < confidence:
                    self.probabilities[i] = 0
                    self.pre_preds[i] = '0'
            if self.pre_preds[i] in ['13', '14']:
                if self.probabilities[i] < 0.85:
                    self.probabilities[i] = 0
                    self.pre_preds[i] = '0'
    def run(self):
        # slowfast model
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        slowfast_model = load_engine(trt_runtime=trt_runtime, plan_path=self.slowfast_path)

        # yolov5
        device = select_device("")
        yolov5_model = DetectMultiBackend(self.yolov5_path, device=device, dnn=False, data=self.gesture_data_yaml, fp16=False)
        stride, names_1, pt = yolov5_model.stride, yolov5_model.names, yolov5_model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        yolov5_model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup

        yolov5_model_1 = DetectMultiBackend(self.yolov5_face_path, device=device, dnn=False, data=self.face_data_yaml, fp16=False)
        stride_1, names_2, pt_1 = yolov5_model_1.stride, yolov5_model_1.names, yolov5_model_1.pt
        names_2[0] = "head"
        names_2[1] = "hand"
        yolov5_model_1.warmup(imgsz=(1 if pt_1 else 1, 3, *imgsz))  # warmup

        batch_size = 1
        maxsize = 32

        self.isStart = []
        self.timeframes = []
        self.frames = []
        self.probabilities = []
        self.pre_preds = []
        cls = lambda x: x if x in self.conf_yolov5 else "others"  # 获取在conf_yolov5中的名称

        flag = 0

        repeat_num = 0
        repeat = 0
        while True:
            frame, timeframe = self.pipe.recv()
            annotator = Annotator(frame, line_width=self.line_thickness, example=str(names_1))
            im0 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            bbx1, annotator = self.inference(im0=im0,
                                             yolov5_model=yolov5_model,
                                             names=names_1,
                                             view_img=view_img and self.show_gesture,
                                             annotator=annotator,
                                             device=device,
                                             pred_conf=self.conf_yolov5)
            bbx2, annotator = self.inference(im0=im0,
                                             yolov5_model=yolov5_model_1,
                                             names=names_2,
                                             view_img=view_img and self.show_head,
                                             annotator=annotator,
                                             device=device,
                                             pred_conf=self.conf_yolov5)
            if len(bbx1) or len(bbx2):
                data1 = bbx1.clone().detach().cpu().tolist() if len(bbx1) else bbx1.copy()
                data2 = bbx2.clone().detach().cpu().tolist() if len(bbx2) else bbx2.copy()

                data = []
                data.extend(self.getData(data1, names_1))
                data.extend(self.getData(data2, names_2))

                self.main_pipe.send({
                    "model": "yolov5",
                    "data": {
                        "bbxes": data
                    }
                })


            # show images
            frame = annotator.result()
            if view_img:
                cv.imshow('video', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                    break

            if len(bbx1) and flag == 0:
                max_idx = torch.argmax(bbx1[:, 4], dim=0).item() if len(bbx1) else 0
                if len(bbx1) and int(bbx1[max_idx][5]) <= 12 and \
                        round(bbx1[max_idx][4].item(), self.decimal_precision) > self.conf_yolov5[cls(int(bbx1[max_idx][5]))] and \
                        names_1[int(bbx1[max_idx][5])] in self.start_frame_clses:
                    flag = 1
                    repeat = int(bbx1[max_idx][5])


            if flag == 1:
                self.frames.append(frame)
                self.timeframes.append(timeframe)
                self.isStart.append(len(bbx1))
                max_idx = torch.argmax(bbx1[:, 4], dim=0).item() if len(bbx1) else 0
                self.probabilities.append(round(bbx1[max_idx][4].item(), self.decimal_precision) if len(bbx1) else 0)  # yolov5 概率
                self.pre_preds.append(names_1[int(bbx1[max_idx][5])] if len(bbx1) and int(bbx1[max_idx][5]) <= 12 else '0')  # yolov5 預測
                if len(bbx1) and int(bbx1[max_idx][5]) == repeat:
                    repeat_num += 1
                else:
                    repeat = -1
                if repeat_num == 10:
                    self.frames.pop()
                    self.timeframes.pop()
                    self.isStart.pop()
                    self.probabilities.pop()
                    self.pre_preds.pop()
                    repeat_num -= 1

            else:
                continue


            if len(self.frames) >= maxsize and self.isStart[0] \
                and self.probabilities[0] > self.conf_yolov5[cls(self.pre_preds[0])] \
                and self.pre_preds[0] in self.start_frame_clses:


                ######################################################################################################

                self.filter_pred_prob(confidence = 0.9)

                pred_of_yolov5 = self.pre_preds[:32]
                start = self.judge_start(pred_of_yolov5,num = 10)

                if start:

                    inputs = get_frames(self.cfg, self.frames)
                    h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream = allocate_buffers(slowfast_model, batch_size, trt.float32)

                    out = do_inference(slowfast_model, inputs, h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream)
                    prob = self.softmax(out,self.decimal_precision)
                    pred = np.argmax(out)

                ################################################################

                    if self.verbose:
                        print(pred, prob)
                    if self.debug:
                        curr_time = time.time()
                        a = torch.tensor(inputs[1].clone().detach()).squeeze(0).transpose(0, 1)
                        torchvision.utils.save_image(a, f'./debug/{curr_time}-{pred}.png')

                    self.main_pipe.send({
                        "model": "slowfast",
                        "data": {
                            "pred_of_yolov5": self.pre_preds[:32],
                            "prob_of_yolov5": self.probabilities[:32],
                            "pred_of_slowfast": pred,
                            "prob_of_slowfast": prob,
                            "prob_without_softmax":out
                        }
                    })
                flag = 0

                ################################################################
                self.frames.clear()
                self.timeframes.clear()
                self.isStart.clear()
                self.probabilities.clear()
                self.pre_preds.clear()
                repeat_num = 0

    @staticmethod
    def softmax(logits,decimal_precision):
        wanted = [1,2,3,4,11,12,13]
        wanted_logits = [logits[i] for i in wanted]
        e_x = np.exp(wanted_logits)
        temp = e_x / np.sum(e_x, axis=-1, keepdims=True)
        probs = []
        i = 0
        for id in range(len(logits)):
            if id not in wanted:
                probs.append(0)
            else:
                probs.append(round(temp[i],decimal_precision))
                i+=1
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
    camera = CameraProcess(xname='camera',
                           pipe=p1)  # 和model进程通信的双工管道
    model = ModelProcess(xname='model',
                         pipe=p2,  # 和camera进程通信的双工管道
                         main_pipe=p4,  # 和main进程通信的单工管道
                         slowfast_path="../engines/slowfast_rgb.plan",  # 视频理解模型
                         yolov5_path="../checkpoints/yolov5m_4.pt",  # 手势检测模型
                         yolov5_face_path="../checkpoints/face.pt",  # 头手检测模型
                         cfg=cfg,  # slowfast 配置文件
                         debug=False,  # 是否进入debug模式
                         show_video=True,  # 是否展现摄像头
                         show_head=True,  # 是否展现头手
                         show_gesture=True,  # 是否展现手势
                         decimal_precision=4,  # 小数精度
                         conf_yolov5={"head": 0.4, "hand": 0.2, "others": 0.7})  # 头和手置信度
    camera.start()
    model.start()

    while True:
        data = p3.recv()
        model = data["model"]
        if model == 'slowfast':
            print(data["data"]['pred_of_yolov5'])
            print(data["data"]['prob_of_yolov5'])
            print(data["data"]['pred_of_slowfast'])
            print(data["data"]['prob_of_slowfast'])
        elif model == 'yolov5':
            print(data["data"]["bbx"])


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)

    mytest_trt(cfg=cfg)
