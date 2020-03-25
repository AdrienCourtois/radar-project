import torch
import numpy as np

def focale_loss(y_pred, y_true, alpha=0.75, gamma=2):
    # y_pred: tensor [B, 1, H, W] binary predicted mask
    # y_true: tensor [B, 1, H, W] binary ground truth mask
    # --
    # Output: tensor [B] loss for each prediction of the batch

    m1 = y_true == 1
    m0 = y_true == 0

    p_t = torch.zeros(y_pred.size())
    alpha_t = torch.zeros(y_pred.size())

    if y_pred.is_cuda:
        alpha_t = alpha_t.cuda()
        p_t = p_t.cuda()

    p_t[m1] = y_pred[m1]
    p_t[m0] = 1-y_pred[m0]

    alpha_t[m1] = alpha
    alpha_t[m0] = 1-alpha

    L = - alpha_t * ((1 - p_t) ** gamma) * torch.log(p_t + 1e-10)

    return L.sum((1,2,3))

def iou_loss(y_pred, y_true):
    b_iou = torch.zeros(y_true.size(0))
    if y_pred.is_cuda:
        b_iou = b_iou.cuda()
    
    for i in range(y_true.size(0)):
        num, denom = None, None

        if y_true[i].sum() == 0:
            num = ((1-y_pred[i]) * (1-y_true[i])).sum()
            denom = (1-y_pred[i] + 1-y_true[i] - (1-y_pred[i]) * (1-y_true[i])).sum()
        else:
            num = (y_pred[i] * y_true[i]).sum()
            denom = (y_pred[i] + y_true[i] - y_pred[i] * y_true[i]).sum()

        b_iou[i] = num / denom

    return b_iou[i]