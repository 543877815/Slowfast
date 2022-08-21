import torch
# import numpy as np
# def AUC_my(label, pred):
#     """
#     label是gt
#     pred是对pos的预测结果
#     计算方法：计算任意一对真正样本和真负样本的概率比较，如果真正样本 > 真负样本，则0
#     如果 如果真正样本 < 真负样本： 累计1个惩罚
#     如果 如果真正样本 = 真负样本：累计0.5个惩罚
#     此计算的是曲线上面的面积，1-a 才是auc的结果
#     """
#     auc = 0.0
#     pos = []
#     neg = []
#     for i, s in enumerate(label):
#         if s == 1:
#             pos.append(i)
#         else:
#             neg.append(i)
#     for i in pos:
#         for j in neg:
#             if pred[i] > pred[j]:
#                 auc += 1
#             elif pred[i] == pred[j]:
#                 auc += 0.5
#
#     return auc / (len(pos) * len(neg))


################################# nms  ###################
def nms(preds, scores, threhold=0.5):
    # pred N 4
    # scores [N]

    x1 = preds[:, 0]
    y1 = preds[:, 1]
    x2 = preds[:, 2]
    y2 = preds[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)  # 返回降序的index

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        # 删除所有与当前框iou超过阈值的框
        # 计算当前框与其他框的iou
        top_left_x = x1[order[1:]].clamp(min=x1[i])  # 较小的
        top_left_y = y1[order[1:]].clamp(min=y1[i])

        bottom_right_x = x2[order[1:]].clamp(max=x2[1])
        bottom_right_y = x2[order[1:]].clamp(max=y2[1])

        area_inter = (bottom_right_x - top_left_x).clamp(min=0) * (bottom_right_y - top_left_y).clamp(min=0)

        iou = area_inter / (areas[order[1:]] + areas[i] - area_inter)
        idx = (iou <= threhold).nonzero().squeeze()

        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 保留与iou阈值小的框
    return torch.LongTensor(keep)


if __name__ == '__main__':
    # label = [1, 0, 0, 0, 1, 0, 1, 0]
    # pre = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
    # print(AUC_my(label, pre))
    #
    # from sklearn import metrics
    # auc = metrics.roc_auc_score(label, pre)
    # print('sklearn', auc)

    preds = torch.tensor([
        [0, 0, 4, 4],
        [1, 1, 5, 5],
        [2, 2, 3, 3],
        [4, 4, 8, 8],
        [3, 3, 8, 8]
    ])

    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.8])

    print(nms(preds, scores))