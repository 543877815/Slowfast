import os

video_type = "RGB"

os.system("python vid2img.py /data/lifengjun/gesture_dataset/dataset/video/{}/train /data/lifengjun/gesture_dataset/dataset/image/{}/train".format(video_type, video_type))
os.system("python vid2img.py /data/lifengjun/gesture_dataset/dataset/video/{}/val /data/lifengjun/gesture_dataset/dataset/image/{}/val".format(video_type, video_type))
os.system("python vid2img.py /data/lifengjun/gesture_dataset/dataset/video/{}/test /data/lifengjun/gesture_dataset/dataset/image/{}/test".format(video_type, video_type))

os.system("python generate_annotation.py --video_type {}".format(video_type))