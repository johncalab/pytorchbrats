"""
Convert .nii.gz files to numpy.
Starts from dataPath and expects Task01_BrainTumour folder, containing imagesTr and labelsTr.
Creates folders with numpy files.
"""
import os
import nibabel as nib
import numpy as np
import skimage.transform
import tqdm

# Parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type=str, default='/rsrch1/ip/jrcalabrese')
parser.add_argument('-r', type=bool, help='If True, it will downsample images.', default=True)
parser.add_argument('-size', type=int, help="If -r is True, it will downsample to -size.", default=32)
parser.add_argument('-low', type=int, help="Lower slice bound.", default=46)
parser.add_argument('-high', type=int, help="Upper slice bound.", default=110)
parser.add_argument('-clip', type=bool, help="If True, it will clip labels to 0,1.", default=True)
parser.add_argument('-chanFirst', type=bool, help="If True, puts channel axis first.", default=True)
parser.add_argument('-cxlow', type=int, help="Crop lower bound on x-axis", default=50)
parser.add_argument('-cxhigh', type=int, help="Crop higher bound on x-axis", default=195)
parser.add_argument('-cylow', type=int, help="Crop lower bound on y-axis", default=20)
parser.add_argument('-cyhigh', type=int, help="Crop higher bound on y-axis", default=200)
args = parser.parse_args()

dataPath = args.dataPath
RESIZE = args.r
SIZE = args.size
SLICE_LOW = args.low
SLICE_HIGH = args.high
CLIP = args.clip
CHANFIRST = args.chanFirst
CROP_X_LOW = args.cxlow
CROP_X_HIGH = args.cxhigh
CROP_Y_LOW = args.cylow
CROP_Y_HIGH = args.cyhigh

print(f'dataPath is {dataPath}')

# before we start, a helper function to resize images
def img_resize(img,
    lower=SLICE_LOW,
    upper=SLICE_HIGH,
    xlower=CROP_X_LOW,
    xupper=CROP_X_HIGH,
    ylower=CROP_Y_LOW,
    yupper=CROP_Y_HIGH,
    size=SIZE,
    inp=True,
    clip=False):
    """
    img is a 3d or 4d numpy array: DxDxSxC
        - D is dimension of width and height
        - S is the number of slices
        - C is the number of channels
    0 <= lower < upper <= S
    0 <= size <= D
    
    if clip = True, it groups all non-background together.
    """
    if inp:
        img = img[xlower:xupper,ylower:yupper,lower:upper,:]
        slices = img.shape[2]
        channels = img.shape[3]
        img = skimage.transform.resize(img,(size,size,slices,channels))
        return img
    else:
        img = img[xlower:xupper,ylower:yupper,lower:upper]
        slices = img.shape[2]
        img = skimage.transform.resize(img,
                                      (size,size,slices),
                                      preserve_range=True,
                                      anti_aliasing=False,
                                      order=0)
        img = img.astype('uint8')
        if clip:
            img = np.clip(img,0,1)
        return img


# Here we go ---------------------
# obtain paths to data and perform sanity check
trainpath = os.path.join(dataPath, 'Task01_BrainTumour', 'imagesTr')
labelspath = os.path.join(dataPath, 'Task01_BrainTumour', 'labelsTr')

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

# input data
if RESIZE:
    newPath = os.path.join(dataPath, 'numpyData'+str(SIZE))
else:
    newPath = os.path.join(dataPath, 'numpyDataOG')
if not os.path.exists(newPath):
    os.mkdir(newPath)

numtrainpath = os.path.join(newPath, 'source')
if os.path.exists(numtrainpath):
    print('Folder already exists, so I did not create any numpy from train.')
else:
    os.mkdir(numtrainpath)

    print('I am starting to convert training images.')
    path = trainpath
    trainprogress = tqdm.tqdm(enumerate(trainlocations))
    for i, imageLocation in trainprogress:
        trainprogress.set_description(f"Processing image {imageLocation}")
        # get the .nii image
        imageData = nib.load(os.path.join(path,imageLocation))
        # convert to numpy
        numpyImage = imageData.get_data()
        if RESIZE:
            numpyImage = img_resize(numpyImage)
        if CHANFIRST:
            numpyImage = np.transpose(numpyImage,(3,0,1,2))
        np.save(os.path.join(numtrainpath, imageLocation), numpyImage)

# output data
numlabelspath = os.path.join(newPath, 'target')
if os.path.exists(numlabelspath):
    print('Folder alreay exists, so I did not create any numpy files from labels.')
else:
    os.mkdir(numlabelspath)

    print('I am starting to convert labels images.')
    path = labelspath
    labelsprogress = tqdm.tqdm(enumerate(labelslocations))
    for i, imageLocation in labelsprogress:
        labelsprogress.set_description(f"Processing image {imageLocation}")
        # get the .nii image
        imageData = nib.load(os.path.join(path, imageLocation))
        # convert to numpy
        numpyImage = imageData.get_data()
        # resize
        if RESIZE:
            numpyImage = img_resize(numpyImage,inp=False,clip=CLIP)
        # save
        np.save(os.path.join(numlabelspath,imageLocation), numpyImage)

print('All done, I think.')