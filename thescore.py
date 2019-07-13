import torch
import torch.nn as nn
import numpy as np

def iou_loss(y_pred, y, SMOOTH=1e-6):
    """
    essentially returns 1 - iou_score
    but takes care of y_pred not being rounded
    """
    assert y_pred.shape == y.shape
    axes = tuple([i for i in range(1,len(y.shape))])
    batch_len = y.shape[0]
    
    numerator = y*y_pred
    numerator = numerator.sum(dim=axes)
    a = y.sum(dim=axes)
    b = y.sum(dim=axes)
    denominator = a + b - numerator
    quotient = 1 - ((numerator + SMOOTH) / (denominator + SMOOTH))
    return quotient.mean()


class iouModule(nn.Module):
    def __init__(self):
        super(iouModule,self).__init__()

    def forward(self, y_pred, y):
        loss = iou_loss(y_pred, y)
        return loss
    
def iou_score(y_pred, y, SMOOTH=1e-6):
    """
    aka Jaccard
    expect: y_pred, y to be of same INTEGER type
    
    y_pred is output of model
        expect: y_pred.shape = (batch_len,D,D,S)
        (no channels!)
    y is truth value (labels)
        expect: y.shape = (batch_len,D,D,S)
    
    returns: the mean across the batch of the iou scores
    """
    # sanity check
    assert y_pred.shape == y.shape
    # to compute scores, we sum along all axes except for batch
    axes = tuple([i for i in range(1,len(y.shape))])
    batch_len = y.shape[0]
    
    intersection = (y_pred & y).sum(dim=axes).float()
    union = (y_pred | y).sum(dim=axes).float()
    # sanity check
    assert intersection.shape == union.shape
    assert union.shape == (batch_len,)
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return iou.mean()

