import numpy as np
import shutil
import os
import argparse

if __name__ == "__main__":
    ## 这里写的逻辑是将source的文件转成mp4，在target中存放
    ## 当vflip==True的时候实现的是水平翻转
    parser = argparse.ArgumentParser()
    parser.add_argument("--vflip", default=False, type=bool, help="whether to flip the video")
    parser.add_argument("--source", default=r"H:\gesture_project\dataset\video\RGB1\1", type=str, help="source path of raw video")
    parser.add_argument("--target", default=r"H:\gesture_project\dataset\video\RGB\1", type=str, help="target path of mp4 video")
    args = parser.parse_args()
    vflip = args.vflip
    source_path = r"{}".format(args.source)
    target_path = r"{}".format(args.target)
    os.makedirs(target_path, exist_ok=True)
    for file in os.listdir(source_path):
        source = os.path.join(source_path, file)
        portion = os.path.splitext(file)
        target_name = "v_{}.mp4".format(portion[0]) if vflip else "{}.mp4".format(portion[0])
        target = os.path.join(target_path, target_name)

        ## 如果是保存为avi则使用： ffmpeg -i 030001.avi -vf "hflip" -c:v libx264 -crf 18 -y output.avi
        ## 如果是保存为mp4则使用： ffmpeg -i 030001.avi -vf "hflip" -y output.mp4
        if vflip:
            order = f"ffmpeg -i {source} -qscale 1 -vf 'hflip' -y {target}"
        else:
            order = f"ffmpeg -i {source} -y -qscale 1 {target}"
        os.system(order)
        print(order)