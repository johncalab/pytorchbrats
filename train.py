"""
it's time to train...
"""
from themodel import SmallU
from themodel import dice_loss
import torch
import torch.nn as nn
from torch.utils import Dataset
import os

class myBrats(Dataset):
	"""
	__init__ needs a rootPath
	from there it assumes there are two folders numtrain and numlabels containing numpy files
	"""
    def __init__(self, rootPath):
        self.train = []
        self.labels = []
        
        imgPaths = os.listdir(rootPath + '/numtrain')
        for path in imgPaths:
            if path.endswith('.npy'):
                self.train.append(rootPath + '/numtrain/' + path)
        
        imgPaths = os.listdir(rootPath + '/numlabels')
        for path in imgPaths:
            if path.endswith('.npy'):
                self.labels.append(rootPath + '/numlabels/' + path)
        
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

# dataset
# dataloading
# train
