import torch
import numpy as np

def iouscore(ypred,y,threshold=0.5):
    'Expects ypred and y to be torch tensors'
    ypred = ypred.detach()
    ypred = torch.sigmoid(ypred)
    # might have to change this to ypred.cpu().numpy()
    ypred = ypred.cpu().numpy()
    ypred = (ypred > threshold).astype(int)
    
    #might have to change this to y = y.detach().cpu().numpy().astype(int)
    y = y.detach().cpu().numpy().astype(int)
    
    intersection = np.logical_and(ypred,y)
    union = np.logical_or(ypred,y)
    score = np.sum(intersection)/np.sum(union)
    return score