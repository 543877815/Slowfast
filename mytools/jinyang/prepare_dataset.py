import os

video_type = "RGB"

os.system("python vid2img.py /data/lifengjun/data/gesture_dataset/mp4/train/ /data/lifengjun/data/gesture_dataset/images/train/".format(video_type, video_type))
os.system("python vid2img.py /data/lifengjun/data/gesture_dataset/mp4/val/ /data/lifengjun/data/gesture_dataset/images/val/".format(video_type, video_type))
# os.system("python vid2img.py /data/lifengjun/gesture_dataset/dataset/video/{}/test /data/lifengjun/gesture_dataset/dataset/image/{}/test".format(video_type, video_type))

os.system("python generate_annotation.py --video_type {}".format(video_type))