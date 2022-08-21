import tensorrt as trt
import argparse
from onnx import ModelProto

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, shape1=[1, 3, 16, 256, 256], shape2=[1, 3, 64, 256, 256]):
    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file.
   """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape1
        network.get_input(1).shape = shape2
        engine = builder.build_engine(network, config)
        return engine

def build_engine1(onnx_path, shape=[1, 3, 256, 256]):
    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file.
   """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


if __name__ == "__main__":
    engine = build_engine('../onnxes/slowfast_grayscale.onnx')
    save_engine(engine, '../engines/slowfast_grayscale.plan')
    print('finish 1')

    engine1 = build_engine1('../onnxes/mobileNetv2.onnx')
    save_engine(engine1, '../engines/mobileNetv2.plan')
    print('finish 2')
