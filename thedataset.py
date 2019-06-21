"""
We have converted our .nii.gz files into numpy.
This dataset locates the files and provides a method to retrieve them: np.load + convert to tensor.
Training will then use this dataset to train the U-net.
"""
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class bratsDataset(Dataset):
    """
    __init__ needs a rootPath
    from there it assumes there are two folders numtrain and numlabels containing numpy files
    """
    def __init__(self, rootPath):
        self.train = []
        self.labels = []
        
        pathTrain = os.path.join(rootPath, 'numtrain')
        imgPaths = os.listdir(pathTrain)
        for path in imgPaths:
            if path.endswith('.npy'):
                self.train.append(os.path.join(pathTrain, path))
        
        pathLabels = os.path.join(rootPath, 'numlabels')
        imgPaths = os.listdir(pathLabels)
        for path in imgPaths:
            if path.endswith('.npy'):
                self.labels.append(os.path.join(pathLabels, path))
        
    def __len__(self):
        assert len(self.train) is len(self.labels)
        return len(self.train)
    
    def __getitem__(self,idx):
        x = np.load(self.train[idx])
        x = torch.from_numpy(x)
        x = x.float()
        
        y = np.load(self.labels[idx])
        y = torch.from_numpy(y)
        y = y.float()
        
        return x,y

