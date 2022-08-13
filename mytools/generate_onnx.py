import os
import sys

sys.path.insert(0, '../')
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
import torch
import onnx
from onnxsim import simplify

def generate_onnx(cfg):
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()

    batch_size = 1
    img_channel = 3
    fast_frame = 16
    slow_frame = 64
    img_width = 256
    img_height = 256
    inputs = [torch.randn(batch_size, img_channel, fast_frame, img_width, img_height, device="cuda"),
              torch.randn(batch_size, img_channel, slow_frame, img_width, img_height, device="cuda")]

    input_names = ["x0", "x1", "bboxes"]
    output_names = ["output"]

    saved_name = "../onnxes/slowfast_rgb.onnx"
    torch.onnx.export(model, args=(inputs[0], inputs[1], None), f=saved_name, verbose=True, input_names=input_names, output_names=output_names)
    print("finish exporting onnx")

    # onnx_model = onnx.load(saved_name)
    # simplified_name = "../onnxes/slowfast_rgb_simp.onnx"
    # model_simp, check = simplify(onnx_model)
    # onnx.save(model_simp, simplified_name)
    # print('finished exporting simplified onnx')


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=generate_onnx)
