"""
Write a docstring later.
"""
import numpy as np
import os
import nibabel as nib
from scipy import ndimage
import skimage.transform

rootPath = './playData/'

# Check how many images we have:
NUM_SAMPLES = len(os.listdir(rootPath + 'X'))
assert NUM_SAMPLES is len(os.listdir(rootPath + 'Y'))

# RESAMPLE_SIZE is really small for now to speed things up

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8
RESAMPLE_SIZE = 32
NUM_SLICES = 155

# Get paths to data

imgPaths = os.listdir(rootPath + 'Y')
Xlocations = []
for path in imgPaths:
    if path.endswith('.nii.gz'):
        Xlocations.append(path)

imgPaths = os.listdir(rootPath + 'Y')
Ylocations = []
for path in imgPaths:
    if path.endswith('.nii.gz'):
        Ylocations.append(path)

assert Xlocations == Ylocations
assert len(Xlocations) == NUM_SAMPLES


################ 
# Save Training Data
################

# If you want to check whether the numpy file already exists:
# os.path.exists(rootPath + 'X.npy')


path =  rootPath + 'X'
X = np.empty(shape=(NUM_SAMPLES,RESAMPLE_SIZE,RESAMPLE_SIZE,NUM_SLICES), dtype=IMG_DTYPE)

for i, imageLocation in enumerate(Xlocations):
    imageData = nib.load(path + '/' + imageLocation)
    numpyImage = imageData.get_data().astype(IMG_DTYPE)[:,:,:,0]
    
    assert numpyImage.shape[2] == NUM_SLICES
    
    resImage=skimage.transform.resize(numpyImage,(RESAMPLE_SIZE,RESAMPLE_SIZE,NUM_SLICES),order=0,mode='constant',preserve_range=True).astype(IMG_DTYPE)
    X[i] = resImage

np.save(rootPath + 'X', X)

# if you want to load pre-made file
# X = np.load(rootPath + 'X.npy')


################ 
# Save Labels Data
################




# Save Labels Data

# If you want to check whether the numpy file already exists:
# os.path.exists(rootPath + 'Y.npy')

path = rootPath + 'Y'
Y = np.empty(shape=(NUM_SAMPLES,RESAMPLE_SIZE,RESAMPLE_SIZE,NUM_SLICES), dtype=IMG_DTYPE)

for i, imageLocation in enumerate(Ylocations):
    imageData = nib.load(path + '/' + imageLocation)
    numpyImage = imageData.get_data().astype(IMG_DTYPE) # [:,:,:,0] don't need this last part as it's a 3D image
    
    assert numpyImage.shape[2] == NUM_SLICES
    
    resImage=skimage.transform.resize(numpyImage,(RESAMPLE_SIZE,RESAMPLE_SIZE,NUM_SLICES),order=0,mode='constant',preserve_range=True).astype(IMG_DTYPE)
    Y[i] = resImage

np.save(rootPath + 'Y', Y)

# if you want to load pre-made file
# Y = np.load(rootPath + 'Y.npy')

