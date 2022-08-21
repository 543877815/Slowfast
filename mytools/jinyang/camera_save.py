from multiprocessing import Process
import cv2 as cv
import os


class CameraProcess(Process):
    def __init__(self, xname):
        super(CameraProcess, self).__init__()
        self.xname = xname

        self.width = 256
        self.height = 256

    def run(self):
        # 打开摄像头
        self.cap = cv.VideoCapture(0)
        print("camera ready!")

        print(self.cap.set(cv.CAP_PROP_AUTO_WB, 0)) # 关闭白平衡，解决偏色问题
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
            cv.imwrite('./images_raw/{}.png'.format(str(id).zfill(5)), frame)
            # frame = cv.resize(frame, (self.width, self.height), cv.INTER_NEAREST)
            cv.imshow('video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break

            cv.imwrite('./images_resize/{}.png'.format(str(id).zfill(5)), frame)
            id += 1

            # 监测键盘输入是否为q，为q则退出程序
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                break

if __name__ == '__main__':
    os.makedirs('./images_resize', exist_ok=True)
    os.makedirs('./images_raw', exist_ok=True)
    camera = CameraProcess('camera')
    camera.start()
