"""
Assumes data is contained in a dataPath, and is split up in folders numtrain and numlabels.
make_npy.py is not called
"""
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataPath', type=str, default='data', help="dataPath")
parser.add_argument('-bs', '--batchSize', type=int, default=16, help='BatchSize')
parser.add_argument('-ne', '--numEpochs', type=int, default=3, help='NumEpochs')
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold of sigmoid output')
parser.add_argument('--cuda', type=bool, default=False, help='use cuda True/False')
# add train/valid split
# add learning rate? optimizer? other models?
args = parser.parse_args()
# Declare a few global variables.
dataPath = args.dataPath
BATCH_SIZE = args.batchSize
NUM_EPOCHS = args.numEpochs
THRESHOLD = args.threshold
SPLIT_FRAC = 0.25
LEARNING_RATE = 1e-4

print(f'dataPath is {dataPath}')
print(f'NumEpochs is {NUM_EPOCHS}')
print(f'Threshold is {THRESHOLD}')
print(f'BatchSize is {BATCH_SIZE}')
print(f'SPLIT_FRAC is {SPLIT_FRAC}')

"""
In parser, there should be options for: training from scratch, resume training, testing
"""
