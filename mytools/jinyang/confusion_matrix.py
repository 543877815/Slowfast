import sys

sys.path.insert(0, '../')
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

logger = logging.get_logger(__name__)


def generate_cm(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0
    )

    # # Create meters for multi-view testing.
    # test_meter = TestMeter(
    #     test_loader.dataset.num_videos
    #     // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
    #     cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
    #     cfg.MODEL.NUM_CLASSES
    #     if not cfg.TASK == "ssl"
    #     else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
    #     len(test_loader),
    #     cfg.DATA.MULTI_LABEL,
    #     cfg.DATA.ENSEMBLE_METHOD,
    # )
    #
    # writer = None

    model.eval()

    y_pred = []
    y_true = []
    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(
            test_loader
    ):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Perform the forward pass.
        with torch.no_grad():
            preds = model(inputs)
        classes = torch.argmax(preds, 1)

        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()
            classes = classes.cpu()

        y_true.append(labels)
        y_pred.append(classes)
    y_trues = torch.cat(y_true)
    y_preds = torch.cat(y_pred)

    y_true_np = np.array(y_trues)
    y_pred_np = np.array(y_preds)

    correct = np.sum(y_true_np == y_pred_np)
    total = len(y_true_np)

    print("correct rate: {}".format(correct / total))
    C = confusion_matrix(y_true_np, y_pred_np, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    plt.savefig('./confusion-matrix.png')


if __name__ == "__main__":
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)

    launch_job(cfg=cfg, init_method=args.init_method, func=generate_cm)
