import jittor as jt
from jittor import nn
from jittor import Module

from .loss import OhemCrossEntropy2d
from utils.pyt_utils import CrossEntropyLoss

class CriterionDSN(Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True):  # , reduction='mean'
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = CrossEntropyLoss(ignore_index=ignore_index)  # , reduction=reduction
        # if not reduction:
        #     print("disabled the reduction.")

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:  # 上采样？分辨率
            scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = nn.interpolate(preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss

class CriterionOhemDSN(Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True):  # , reduction='mean'
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = CrossEntropyLoss(ignore_index=ignore_index)  # , reduction=reduction

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)  # scale_pred [1,19,769,769,], target [1,769,769,]

        scale_pred = nn.interpolate(preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4
