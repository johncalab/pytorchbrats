"""
To be used when training a model from scratch.
"""
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, random_split
import matplotlib.pyplot as plt
import tqdm
import datetime

# parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dp', type=str, default='ignore/data')
parser.add_argument('-ne', type=int, default=4)
parser.add_argument('-bs', type=int, default=8)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-resolution', type=str, default='32', help='Which resolution to use.')
parser.add_argument('-trainsplit', type=float,default=0.25)
parser.add_argument('-v', type=bool, default=False)
parser.add_argument('-model', type=str, default='Crush')
parser.add_argument('-loss', type=str, default='iou')
parser.add_argument('-dataset', type=str, default='brats3dDataset')
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-plot', type=bool,default=True)
parser.add_argument('-log', type=bool,default=True)

args = parser.parse_args()

# start log file
def gettime():
    now = datetime.datetime.now()
    return now.strftime('%y%m%d_%H%M%S')

start_time = gettime()
logPath = os.path.join('models', start_time + '.log')
from randoname import randoname
rn = randoname()
with open(logPath, 'a') as f:
    f.write(rn + ' started training.')

# global variables for later use
NUM_EPOCHS = args.ne
print(f"NumEpochs is {NUM_EPOCHS}.")
BATCH_SIZE = args.bs
print(f"BatchLength is {BATCH_SIZE}.")
LR = args.lr
print(f"Learning rate is {LR}.")
MOMENTUM = args.momentum
print(f"Momentum is {MOMENTUM}.")
RESOLUTION = args.resolution
print(f"Resolution is {RESOLUTION}.")
if RESOLUTION == '0':
    dataPath = os.path.join(args.dp, 'numpyDataOG')
else:
    dataPath = os.path.join(args.dp, 'numpyData'+RESOLUTION)
print(f"Data path is {dataPath}.")
SPLIT_FRAC = args.trainsplit
print(f"TrainValid split is {SPLIT_FRAC}.")
VERBOSE = args.v
print(f"Verbose is {VERBOSE}.")

# Get data
import thedataset
datasetname = args.dataset
datasetClass = getattr(thedataset, datasetname)
fullDataset = datasetClass(dataPath)
print(f"There are {len(fullDataset)} images in total.")

# Split into training and validation
valid_size = int(SPLIT_FRAC * len(fullDataset))
train_size = len(fullDataset) - valid_size
train_dataset, valid_dataset = random_split(fullDataset, [train_size, valid_size])
print(f"There are {len(train_dataset)} training images, and {len(valid_dataset)} validation images.")

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# set device
device = 'cpu'
if args.cuda and torch.cuda.is_available():
    device = 'cuda'
print(f"Device used is {device}.")
device = torch.device(device)

# initialize model
"""
Desiderata:
    - feeding parameters to model
    - loading a pretrained model
    - no training, only evaluating
    - choosing optimizer, loss function etc from parser
"""
import themodel
modelname = args.model
modelClass = getattr(themodel, modelname)
model = modelClass()
model.to(device)
print(f"Using model {modelname}.")

# set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# loss function
import thescore
if args.loss == 'iou':
    lossClass = getattr(thescore, args.loss+'Module')
else:
    raise KeyError("Only have iou for now.")
criterion = lossClass()
# criterion = nn.BCEWithLogitsLoss()

# score function
from thescore import iou_score
score_fun = iou_score

# Here starts the training
print("\nAll right, let's do this.")

epochLosses = []
epochScores = []
epochTrainScores = []

try:
    for epoch in range(NUM_EPOCHS):
        print(f'\n-------Epoch {epoch+1}-------')

        # training loop----
        print('\n*Training')
        model.train()
        losses = []
        trainscores = []
        batchloop = tqdm.tqdm(train_dataloader)
        for x,y in batchloop:
            optimizer.zero_grad()
            # use cuda if available
            x = x.to(device)
            y = y.to(device)
            # Forward pass
            y_pred = model(x)
            # Compute loss
            y_pred = torch.sigmoid(y_pred) # remove if using BCEWithLogits
            loss = criterion(y_pred,y)
            # Backward pass
            loss.backward()
            # update
            optimizer.step()

            batchloop.set_description(f"Epoch number {epoch+1}, Loss: {loss.item()}")
            losses.append(loss.item())

            y_pred = y_pred.round()
            y_pred = y_pred.byte()
            y = y.byte()
            score = score_fun(y_pred,y)
            trainscores.append(score)

        avgloss = np.asarray(losses).mean()
        print(f"The average training loss was {avgloss}.")
        epochLosses.append(avgloss)

        avgTrainScore = np.asarray(trainscores).mean()
        print(f"The average training score was {avgTrainScore}.\n")
        epochTrainScores.append(avgTrainScore)

    # -------------

        # validation loop----
        print('\n*Validation')
        model.eval()
        with torch.no_grad():
            scores = []
            validloop = tqdm.tqdm(valid_dataloader)
            for x,y in validloop:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)

                y_pred = torch.sigmoid(y_pred)
                y_pred = y_pred.round()
                y_pred = y_pred.byte()
                y = y.byte()
                score = score_fun(y_pred,y)
                scores.append(score)
            
            avgscore = np.asarray(scores).mean()
            epochScores.append(avgscore)
            print(f"The average score was {avgscore}.")


            # save/overwrite losses and scores
            np.save(os.path.join('models','losses'), epochLosses)
            np.save(os.path.join('models','scores'),epochScores)
except KeyboardInterrupt:
    torch.save(model.state_dict(), os.path.join('models','model.pt'))
    print(f"Current model saved as model.pt")

print('\nWhile training, these were the mean losses:\n')
print(epochLosses)
print('\nWhile training, these were the mean scores:\n')
print(epochTrainScores)
print('\nWhile validating, these were the mean scores:\n')
print(epochScores)

print("\nI am saving the current model now.")
torch.save(model.state_dict(), os.path.join('models','model.pt'))

if args.plot:
    import matplotlib.pyplot as plt
    plt.figure()

    plt.subplot(211)
    plt.title('Needs a better title.')
    plt.plot(epochLosses,'b', label='Training losses')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochScores,'g', label='Validation scores')
    plt.plot(epochTrainScores,'r', label='Training scores')
    plt.legend()

    plt.show()
    plt.savefig(os.path.join('models','plot.png'))

"""
To reload models: 
    model = myModel()
    model.load_state_dict(torch.load(PATH))
    model.eval()
NOTE: jupyter might print some 'incompatible keys' nonsense. Just ignore.
"""