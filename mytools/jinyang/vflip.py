import numpy as np
import shutil
import os

if __name__ == "__main__":
    ## 这里写的逻辑是将source的文件转成mp4，在target中存放
    ## 当vflip==True的时候实现的是水平翻转
    label = 12
    vflip = True
    source_path = 'H:\\gesture_project\\dataset\\video\\RGB1\\{}'.format(label)
    target_path = 'H:\\gesture_project\\dataset\\video\\RGB\\{}'.format(label)
    os.makedirs(target_path, exist_ok=True)
    for file in os.listdir(source_path):
        source = os.path.join(source_path, file)
        portion = os.path.splitext(file)
        target = os.path.join(target_path, portion[0] + '.mp4')

        ## 如果是保存为avi则使用： ffmpeg -i 030001.avi -vf "hflip" -c:v libx264 -crf 18 -y output.avi
        ## 如果是保存为mp4则使用： ffmpeg -i 030001.avi -vf "hflip" -y output.mp4
        if vflip:
            order = f"ffmpeg -i {source} -vf 'hflip' -y {target}"
        else:
            order = f"ffmpeg -i {source} -y {target}"
        os.system(order)
        print(order)