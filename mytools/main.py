import jinyang
from jinyang.camera import ModelProcess, CameraProcess
from slowfast.utils.parser import load_config, parse_args
from multiprocessing import Pipe
import os
import numpy as np
import time
from collections import defaultdict


def getCenter(bbx):
    center_x = (bbx[0] + bbx[2]) / 2
    center_y = (bbx[1] + bbx[3]) / 2
    return [center_x, center_y]


def getFreq(pred_of_yolov5, prob_of_yolov5, pred_of_slowfast, prob_of_slowfast, prob_without_softmax, slowfast_threshold, alpha=1.7, mode="freq"):
    assert mode in ["freq", "medium", "mean"], f"{mode} is not a legal args for mode, option is [\"freq\", \"medium\", \"mean\"]"
    if mode == "freq":
        counter = defaultdict(int)
        count_5 = 0  ##用于计算yolo中出现5的数量
        count_14 = 0
        for i in pred_of_yolov5:
            if i == '0' or i == '13' or i == '14' or i == '5':  # 如果是13(捡东西)或者14(看书)则不计算频数
                if i == '5':
                    count_5 += 1
                if i == '14':
                    count_14 += 1
                continue
            if i == '3':
                counter['4'] += 1  # 如果起始手势是3，3和4的频数都要加一
            if i == '11':
                counter['12'] += 1  # 如果起始手势是11，11和12的频数加一
            counter[i] += 1
        if (count_5 >= 8 or count_14 >= 8) and pred_of_yolov5[0] != '11':  ##如果5出现的太多，说明绝对是误触发，因为其他动作中不会出现大量的5，这时放回无关类
            return 13


        if prob_without_softmax[13] > 0.8 and counter['1'] < 5 and counter['2'] < 5 and counter['3'] < 4 and counter[
            '11'] < 4:  ##如果不使用概率相乘的情况下，无关类的概率特别高，此时要看下起始动作的个数够不够多，如果不够多说明真的是无关类
            return 13

        if prob_without_softmax[13] >= 0.5:
            prob_of_slowfast[13] /= (0.8 + prob_without_softmax[13])  ##如果概率特别高但是起始手势任意一个超过8时，则让无关类概率缩小一点
        # if prob_without_softmax[13] >= 0.5:
        #     prob_of_slowfast[13] /= 1.1  ##如果概率特别高但是起始手势任意一个超过8时，则让无关类概率缩小一点

        for i in range(len(prob_of_slowfast)):  # 频率乘概率
            if str(i) in counter:
                prob_of_slowfast[i] = prob_of_slowfast[i] * (1 + (alpha * (counter[str(i)]) / len(pred_of_yolov5)))
            elif i != 13:  ##无关类的概率保留，其他的只要在yolo的pred中没出现过的，全都概率置为0
                prob_of_slowfast[i] = 0

        for i in range(len(prob_of_slowfast) - 2):
            if i != 3 and i != 4 and prob_of_slowfast[i] < slowfast_threshold:  ##如果经过YOLO加权后的概率还是小于某个阈值，那么就认为是无关类
                prob_of_slowfast[i] = 0

        if pred_of_yolov5[0] != '3':
            for i in range(len(prob_of_slowfast)):  ##这里是判断即使某一类的概率很大，但如果yolo的所有预测里，该手势出现的很少，那也不认为分类正确了(3和4除外，因为3和4在运动中识别不出来，后面如果识别出来了可能需要改)
                if i != 13:
                    if i == 11 or i == 12:
                        yolo_threshold = 3
                    else:
                        yolo_threshold = 5
                    # if prob_of_slowfast[i] < 0.25 and counter[str(i)] < yolo_threshold:
                    if counter[str(i)] < yolo_threshold:
                        prob_of_slowfast[i] = 0

        if np.argmax(prob_of_slowfast) in [3, 4] and (counter['2'] > 5 or counter['1'] > 5):
            prob_of_slowfast[3] = 0
            prob_of_slowfast[4] = 0

        result = np.argmax(prob_of_slowfast)

        if (result == 1 and (counter['11'] > 4)) or (result == 2 and (counter['11'] > 4 or counter['2']<8 or '2' not in pred_of_yolov5[:6])) \
                or (result == 11 and (counter['1'] > 4 or counter['2'] > 4)) or (result == 12 and (counter['1'] > 4 or counter['2'] > 4)) or \
                (result == 3 and (counter['1']>=4 or counter['2']>=4)) or (result == 4 and (counter['1']>=4 or counter['2']>=4)):
            result = 13

        flag1 = flag2 = 1

        if (result == 1):
            flag1 = 0
            flag2 = 0
            prob = prob_of_slowfast[1]
            for i in pred_of_yolov5[12:]:
                if i not in ['0', '13', '14', '5']:
                    flag1 = 1
            for i in pred_of_yolov5[18:]:
                if i != '1':
                    flag2 = 1

        if (result == 2):
            flag1 = 0
            flag2 = 0
            prob = prob_of_slowfast[2]
            for i in pred_of_yolov5[12:]:
                if i not in ['0', '13', '14', '5', '3']:
                    flag1 = 1
            for i in pred_of_yolov5[14:]:
                if i != '2':
                    flag2 = 1

        if flag1 == 0 or flag2 == 0:
            result = 13

        ###########以下待定
        if (result == 3 or result == 4) and counter['3'] > 24:
            result = 13

        if result == 3:
            if prob_without_softmax[4] > prob_without_softmax[3]:
                result = 4
        # print('yolo—pred：', pred_of_yolov5)

        return result

    elif mode == "medium":
        raise NotImplementedError()
    elif mode == "mean":
        raise NotImplementedError()


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
    # https://docs.qq.com/doc/DZndsRG9hQU5qdWFj
    model = ModelProcess(xname='model',
                         pipe=p2,  # 和camera进程通信的双工管道
                         main_pipe=p4,  # 和main进程通信的单工管道
                         slowfast_path="../engines/slowfast_1-2-3-4-11-12.plan",  # 视频理解模型
                         yolov5_path="../checkpoints/yolov5m_6-final.pt",  # 手势检测模型
                         yolov5_face_path="../checkpoints/face_last.pt",  # 头手检测模型
                         face_data_yaml=r"mytools/jinyang/config/face_data.yaml",  # 头手检测模型
                         gesture_data_yaml=r"mytools/jinyang/config/gesture_data.yaml",  # 手势检测模型
                         cfg=cfg,  # slowfast 配置文件
                         debug=False,  # 是否进入debug模式
                         show_video=True,  # 是否展现摄像头
                         show_head=False,  # 是否展现头手
                         show_gesture=True,  # 是否展现手势
                         show_center=True,  # 是否展示中心点
                         decimal_precision=4,  # 小数精度
                         start_frame_clses=["1", "2", "3", "11"],  # 作为slowfast的起始帧的识别的手势的类别
                         conf_yolov5={"head": 0.4, "hand": 0.2, "11": 0.85, "others": 0.85})  # 头和手置信度

    camera.start()
    model.start()

    while True:
        data = p3.recv()
        model = data["model"]  # 数据来源模型: ["slowfast", "yolov5"]
        if model == 'slowfast':
            pred_of_yolov5 = data["data"]["pred_of_yolov5"]
            prob_of_yolov5 = data["data"]['prob_of_yolov5']
            pred_of_slowfast = data["data"]['pred_of_slowfast']
            prob_of_slowfast = data["data"]['prob_of_slowfast']
            prob_without_softmax = data["data"]['prob_without_softmax']
            print(pred_of_yolov5)  # 64帧每帧检测预测结果
            print(prob_of_yolov5)  # 64帧每帧检测预测概率
            print(getFreq(pred_of_yolov5=pred_of_yolov5,
                          prob_of_yolov5=prob_of_yolov5,
                          pred_of_slowfast=pred_of_slowfast,
                          prob_of_slowfast=prob_of_slowfast,
                          prob_without_softmax=prob_without_softmax,
                          alpha=2.2,
                          slowfast_threshold=0.17))  # 64帧视频理解预测结果
            # print(prob_of_slowfast)  # 64帧视频理解预测概率
        elif model == 'yolov5':
            bbxes = data["data"]["bbxes"]

            # print(bbxes)
            # for bbx in bbxes:
            #     if bbx[-1] == 'head':
            #         print(bbx)
                    # print(getCenter(bbx))
            # for bbx in bbxes:  # 获取bounding box的中心点
            #     if bbx[-1] == '14' or bbx[-1] == '13':
            #         print(getCenter(bbx))


if __name__ == "__main__":
    print(jinyang, jinyang.__version__)
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    mytest_trt(cfg=cfg)
