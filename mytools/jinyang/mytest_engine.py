import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import os
from mytools.jinyang.utils import *
from slowfast.utils.parser import load_config, parse_args
from tqdm import tqdm
import torchvision


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

    stream = cuda.Stream()
    return h_input_1, d_input_1, h_input_2, d_input_2, h_output, d_output, stream


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


def do_inference(engine, pics_1, h_input_1, d_input_1, h_input_2, d_input_2, h_output, d_output, stream):
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


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    ## for slowfast model
    slowfast_path = "../engines/slowfast_1-2-3-4-11-12-last.plan"
    slowfast_model = load_engine(trt_runtime, slowfast_path)
    batch_size = 1
    frame_path = r"D:\jupyter\SlowFast\test_imgs\image\11"
    os.makedirs('./debug', exist_ok=True)
    frames = os.listdir(frame_path)
    times = 1
    start_time = time.time()
    for i in tqdm(range(times)):
        for path in frames:
            img_path = os.path.join(frame_path, path)
            path_to_videos = [os.path.join(img_path, img) for img in os.listdir(img_path)]
            inputs = get_frames(cfg=cfg, path_to_videos=path_to_videos)

            h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream = allocate_buffers(slowfast_model, batch_size, trt.float32)
            out = do_inference(slowfast_model, inputs, h_input1, d_input1, h_input2, d_input2, h_output, d_output, stream)

            ### for debug
            a = torch.tensor(inputs[1]).squeeze(0).transpose(0, 1)
            dataframe = time.time()
            classes = np.argmax(out)  # classes.shape = torch.Size([8])
            # torchvision.utils.save_image(a, f'./debug/{dataframe}-{classes}.png')
            print(classes)

    end_time = time.time()
    print(f"total: {end_time - start_time}, avg:{(end_time - start_time) / (len(frames) * times)}")


    ## for mobileNetV2
    # mobilenet_path = "../engines/mobileNetv2.plan"
    # mobilenetv2_model = load_engine(trt_runtime, mobilenet_path)
    #
    # image_dir = r"D:\jupyter\SlowFast\test_imgs\divide\0"
    # imgs = get_images(image_dir)
    # for img in imgs:
    #     h_input1, d_input1, h_output, d_output, stream = allocate_buffers1(mobilenetv2_model, batch_size, trt.float32)
    #     out = do_inference1(mobilenetv2_model, img, h_input1, d_input1, h_output, d_output, stream)
    #     classes = np.argmax(out)
    #     print(classes)