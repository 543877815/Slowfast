import cv2
import os
import shutil
from utils import get_frames, do_inference, load_engine, allocate_buffers, allocate_buffers1, do_inference1
import tensorrt as trt
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np


def sofmax(logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs


if __name__ == "__main__":

    id = 0
    source_path = r"H:\gesture_project\dataset\video\train\IR"
    # for cls in os.listdir(source_path):
    #     video_paths = os.path.join(source_path, cls)
    #     for video in os.listdir(video_paths):
    #         video_path = os.path.join(video_paths, video)

    # video
    video_path = r"H:\gesture_project\video_flow\2.avi"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    writer = cv2.VideoWriter('2.mp4', fourcc, 30, (640, 480), True)

    # model
    mobilenetv2_path = "../../engines/mobileNetv2.plan"
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    mobilenetv2_model = load_engine(trt_runtime=trt_runtime, plan_path=mobilenetv2_path)

    # data
    test_transform = transforms.Compose([
        transforms.transforms.Resize([256, 256]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    batch_size = 1

    # font
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    fontScale1 = 1
    org1 = (30, 30)
    color1 = (255, 0, 0)
    thickness1 = 2

    if cap.isOpened():
        # video.read() 一帧一帧地读取
        # open 得到的是一个布尔值，就是 True 或者 False
        # frame 得到当前这一帧的图像
        open, frame = cap.read()
    else:
        open = False

    id = 0
    while open:
        ret, frame = cap.read()
        # 如果读到的帧数不为空，那么就继续读取，如果为空，就退出
        if frame is None:
            break
        if ret == True:
            # id += 1
            # cv2.imwrite("H:\\gesture_project\\mydivide\\useless\\IR2-{}.png".format(id), frame)
            # break
            q = test_transform(Image.fromarray(frame))
            h_input1, d_input1, h_output, d_output, stream = allocate_buffers1(mobilenetv2_model, batch_size, trt.float32)
            out = do_inference1(mobilenetv2_model, q, h_input1, d_input1, h_output, d_output, stream)
            print(np.argmax(out))
            id += 1
            out = sofmax(out)
            image_1 = cv2.putText(frame, "{}, [{:.2f},{:.2f}]".format(np.argmax(out), out[0], out[1]), org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
            cv2.imshow("video", image_1)

            writer.write(image_1)

            # 这里使用 waitKey 可以控制视频的播放速度，数值越小，播放速度越快
            # 这里等于 27 也即是说按下 ESC 键即可退出该窗口
            if cv2.waitKey(int(1000 / int(fps))) & 0xFF == 27:
                break
            if id == 600:
                break
    writer.release()
