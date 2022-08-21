import numpy as np
import cv2
import os
from tqdm import tqdm

def a():
    img_h, img_w = 256, 256  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []

    for mode in ['train', 'test']:
        #     for cls in range(1, 14):
        cls = 1
        path = f'/data/lifengjun/gesture_dataset/dataset/image/RGB/{mode}/{cls}'
        frames = os.listdir(path)
        for frame in tqdm(frames, desc=f"mode: {mode}, cls: {cls}"):
            imgs_path = f'/data/lifengjun/gesture_dataset/dataset/image/RGB/{mode}/{cls}/{frame}/'
            imgs_path_list = os.listdir(imgs_path)
            len_ = len(imgs_path_list)
            i = 0
            for item in imgs_path_list:
                img = cv2.imread(os.path.join(imgs_path, item))
                img = cv2.resize(img, (img_w, img_h))
                img = img[:, :, :, np.newaxis]
                img_list.append(img)
                i += 1

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))


if __name__ == "__main__":
    img_h, img_w = 256, 256  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []
    cls = 1
    i = 0
    img_path = "D:\jupyter\SlowFast\mytools\images"
    for frame in tqdm(os.listdir(img_path)):
        imgs_path = os.path.join(img_path, frame)
        img = cv2.imread(imgs_path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))