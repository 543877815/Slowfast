import numpy as np
import time
import PIL.Image as Image
import torchvision.transforms as transforms
import slowfast.utils.logging as logging
from slowfast.datasets import utils as utils
import torch
import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
from mytools.utils import *
from slowfast.utils.parser import load_config, parse_args
from tqdm import tqdm

# slowfast
def allocate_buffers(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine.
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32.

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device.
       h_output_1: Output in the host.
       d_output_1: Output in the device.
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_input_2 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(data_type))

    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
    d_input_2 = cuda.mem_alloc(h_input_2.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_input_2, d_input_2, h_output, d_output, stream

# mobilenetv2
def allocate_buffers1(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine.
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32.

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device.
       h_output_1: Output in the host.
       d_output_1: Output in the device.
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))

    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream

def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)

# slowfast
def do_inference(engine, pics_1, h_input_1, d_input_1, h_input_2, d_input_2, h_output, d_output, stream, batch_size=1):
    """
    This is the function to run the inference
    Args:
       engine : Path to the TensorRT engine
       pics_1 : Input images to the model.
       h_input_1: Input in the host
       d_input_1: Input in the device
       h_output_1: Output in the host
       d_output_1: Output in the device
       stream: CUDA stream
       batch_size : Batch size for execution time

    Output:
       The list of output images

    """

    load_images_to_buffer(pics_1[0], h_input_1)
    load_images_to_buffer(pics_1[1], h_input_2)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)
        cuda.memcpy_htod_async(d_input_2, h_input_2, stream)

        # Run inference.
        # context.profiler = trt.Profiler()
        context.execute_async_v2(bindings=[int(d_input_1), int(d_input_2), int(d_output)], stream_handle=stream.handle)
        # context.execute_v2(bindings=[int(d_input_1), int(d_input_2), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output
        return out

# mobilenetv2
def do_inference1(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream):
    """
    This is the function to run the inference
    Args:
       engine : Path to the TensorRT engine
       pics_1 : Input images to the model.
       h_input_1: Input in the host
       d_input_1: Input in the device
       h_output_1: Output in the host
       d_output_1: Output in the device
       stream: CUDA stream
       batch_size : Batch size for execution time

    Output:
       The list of output images

    """

    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        # context.profiler = trt.Profiler()
        context.execute_async_v2(bindings=[int(d_input_1), int(d_output)], stream_handle=stream.handle)
        # context.execute_v2(bindings=[int(d_input_1), int(d_input_2), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output
        return out


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


# 相同的间隔取帧序列
def get_seq_frames(cfg, path_to_videos):
    """
    Given the video index, return the list of sampled frame indexes.
    Args:
        index (int): the video index.
    Returns:
        seq (list): the indexes of frames of sampled from the video.
    """
    num_frames = cfg.DATA.NUM_FRAMES
    video_length = len(path_to_videos)

    seg_size = float(video_length - 1) / num_frames
    seq = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

# 读取视频序列
def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            img = Image.open(image_path)
            imgs.append(img)

        if all(img is not None for img in imgs):
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

# 视频增强
def video_aug(frame):
    TENSOR_TRANSFORMS = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize(mean=[0.615, 0.613, 0.640],
         #                      std=[0.265, 0.264, 0.253])
         transforms.Normalize(mean=[0.504, 0.511, 0.486],
                              std=[0.300, 0.291, 0.286]),
         ])
    for i in range(len(frame)):
        frame[i] = TENSOR_TRANSFORMS(frame[i])

    return frame

# 根据图片返回序列
def get_images(image_dir):
    imgs = []
    WIDTH, HEIGHT = 256, 256
    transform = transforms.Compose([
        transforms.transforms.Resize([WIDTH, HEIGHT]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.615, 0.613, 0.640],
        #                      std=[0.265, 0.264, 0.253])
        transforms.Normalize(mean=[0.504, 0.511, 0.486],
                             std=[0.300, 0.291, 0.286]),
    ])
    for name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, name)
        img = Image.open(image_path)
        img = transform(img)
        imgs.append(img)
    return imgs

# 根据帧序列返回16和64张图像帧
def get_frames(cfg, path_to_videos):
    if isinstance(path_to_videos[0], str):
        path_to_videos.sort()
    seq = get_seq_frames(cfg, path_to_videos)

    frames_PIL = utils.retry_load_images(
        [path_to_videos[frame] for frame in seq],
    )

    frames_PIL = video_aug(frames_PIL)
    frames = torch.stack(frames_PIL, dim=1)

    min_scale, max_scale, crop_size = [cfg.DATA.TEST_CROP_SIZE] * 3
    NUM_SPATIAL_CROPS = cfg.TEST.NUM_SPATIAL_CROPS  # 1
    RANDOM_FLIP = cfg.DATA.RANDOM_FLIP  # False
    INV_UNIFORM_SAMPLE = cfg.DATA.INV_UNIFORM_SAMPLE  # True
    spatial_temporal_idx = 0
    spatial_sample_index = (
            spatial_temporal_idx % NUM_SPATIAL_CROPS
    )

    frames = utils.spatial_sampling(
        frames,
        spatial_idx=spatial_sample_index,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=RANDOM_FLIP,
        inverse_uniform_sampling=INV_UNIFORM_SAMPLE,
    )
    frames = utils.pack_pathway_output(cfg, frames)
    return frames
