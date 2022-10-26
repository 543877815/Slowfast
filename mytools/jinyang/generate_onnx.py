import os
import sys

sys.path.insert(0, '../')
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.models.MobileNet2 import mobilenet_v2
import torch
import onnx
from onnxsim import simplify


def generate_onnx(cfg):
    model = build_model(cfg)
    print(cfg.TEST.CHECKPOINT_FILE_PATH)
    cu.load_test_checkpoint(cfg, model)
    model.eval()

    batch_size = 1
    img_channel = 3
    fast_frame = 8
    slow_frame = 32
    img_width = 256
    img_height = 256
    inputs = [torch.randn(batch_size, img_channel, fast_frame, img_width, img_height, device="cuda"),
              torch.randn(batch_size, img_channel, slow_frame, img_width, img_height, device="cuda")]
    input_names = ["x0", "x1", "bboxes"]
    output_names = ["output"]
    saved_name = "../onnxes/slowfast_1-2-3-4-11-12-new.onnx"

    torch.onnx.export(model, args=(inputs, None), f=saved_name, verbose=True, input_names=input_names, output_names=output_names)
    print("finish exporting slowfast onnx")

    # onnx_model = onnx.load(saved_name)
    # simplified_name = "../onnxes/slowfast_rgb_simp.onnx"
    # model_simp, check = simplify(onnx_model)
    # onnx.save(model_simp, simplified_name)
    # print('finished exporting simplified onnx')
    # def remove_prefix(state_dict, prefix):
    #     f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    #     return {f(key): value for key, value in state_dict.items()}
    #
    # # load classification model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net = mobilenet_v2().to(device)
    # checkpoint = torch.load('../checkpoints/mobileNetv2_model.pth')
    # state_dict = remove_prefix(checkpoint['net'], 'module.')
    # net.load_state_dict(state_dict)
    # inputs = [torch.randn(batch_size, img_channel, img_width, img_height, device="cuda")]
    # input_names = ["x"]
    # output_names = ["output"]
    # saved_name = "../onnxes/mobileNetv2.onnx"
    # torch.onnx.export(net, args=(inputs[0]), f=saved_name, verbose=True, input_names=input_names, output_names=output_names)
    # print("finish exporting mobileNetv2 onnx")


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
    launch_job(cfg=cfg, init_method=args.init_method, func=generate_onnx)
