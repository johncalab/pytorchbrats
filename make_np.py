"""
Convert .nii.gz files into numpy arrays.
Starts from folder dataPath, expects to find a 'train' and 'labels' folders.
In dataPath, will create a 'numtrain' and 'numlabels' folders, storing the numpy arrays.
"""
import numpy as np
import os
import nibabel as nib
from scipy import ndimage
import skimage.transform

# Eventually this should be taken care of by the parser.
# dataPath = os.path.join('ignore', 'playData')
dataPath = 'data'

# Check how many images we have, and performe a sanity check
trainpath = os.path.join(dataPath, 'train')
NUM_SAMPLES = len(os.listdir(trainpath))
labelspath = os.path.join(dataPath, 'labels')
assert NUM_SAMPLES is len(os.listdir(labelspath))

# these values should be taken care of by the parser and should be in some global workflow folder
# I don't think using int16 is a good idea? PyTorch wants floats anyway.
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8
RESAMPLE_SIZE = 64
NUM_SLICES = 155
FINAL_SLICES = 128

# obtain paths to data
imgPaths = os.listdir(trainpath)
trainlocations = []
for path in imgPaths:
    if path.endswith('.nii.gz'):
        trainlocations.append(path)

imgPaths = os.listdir(labelspath)
labelslocations = []
for path in imgPaths:
    if path.endswith('.nii.gz'):
        labelslocations.append(path)

assert trainlocations == labelslocations
assert len(trainlocations) == NUM_SAMPLES

# input data
numtrainpath = os.path.join(dataPath, 'numtrain')
if os.path.exists(numtrainpath):
    print('numtrain already exists, so I did not create any numpy from train')
else:
    os.mkdir(numtrainpath)

    path = trainpath
    for i, imageLocation in enumerate(trainlocations):
        imageData = nib.load(os.path.join(path,imageLocation))
        # we are ignoring all other channels
        numpyImage = imageData.get_data().astype(IMG_DTYPE)[:,:,:,1]

        # sanity check
        assert numpyImage.shape[2] == NUM_SLICES
        resImage=skimage.transform.resize(numpyImage,(RESAMPLE_SIZE,RESAMPLE_SIZE,NUM_SLICES),order=0,mode='constant',preserve_range=True).astype(IMG_DTYPE)
        # throw away bottom slices
        remImage = resImage[:,:,27:]
        # add fake channel
        finalImage = np.expand_dims(remImage, axis=0)

        # sanity check
        assert finalImage.shape[3] == FINAL_SLICES
        # save
        np.save(os.path.join(numtrainpath, imageLocation), finalImage)

# output data
numlabelspath = os.path.join(dataPath, 'numlabels')
if os.path.exists(numlabelspath):
    print('numlabels alreay exists, so I did not create any numpy files from labels')
else:
    os.mkdir(numlabelspath)

    path = labelspath
    for i, imageLocation in enumerate(labelslocations):
        imageData = nib.load(os.path.join(path, imageLocation))
        numpyImage = imageData.get_data().astype(IMG_DTYPE)

        # sanity check
        assert numpyImage.shape[2] == NUM_SLICES
        resImage=skimage.transform.resize(numpyImage,(RESAMPLE_SIZE,RESAMPLE_SIZE,NUM_SLICES),order=0,mode='constant',preserve_range=True).astype(IMG_DTYPE)
        # throw away bottom slices
        remImage = resImage[:,:,27:]
        # add fake channel
        finalImage = np.expand_dims(remImage, axis=0)

        # sanity check
        assert finalImage.shape[3] == FINAL_SLICES
        # save
        np.save(os.path.join(numlabelspath,imageLocation), finalImage)

