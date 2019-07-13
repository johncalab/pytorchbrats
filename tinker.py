import os
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('-datasetname', type=str, default='brats3dDataset')
parser.add_argument('-datapath', type=str, default='ignore/data')
parser.add_argument('-resolution', type=str, help='Resolution size, 0 is for OG', default='32')
args = parser.parse_args()
datasetName = args.datasetname
RESOLUTION = args.resolution
if RESOLUTION == '0':
    dataPath = os.path.join(args.datapath,'numpyDataOG')
else:
    dataPath = os.path.join(args.datapath,'numpyData'+RESOLUTION)

import thedataset
datasetClass = getattr(thedataset, datasetName)
fullDataset = datasetClass(dataPath)