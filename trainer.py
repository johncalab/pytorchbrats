"""
To be used when training a model from scratch.
"""
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
from themodel import SmallU3D
import os
import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import tqdm
import numpy as np
from thescore import iouscore

# parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataPath', type=str, default='ignore/data')
parser.add_argument('-ne', '--numEpochs', type=int, default=4)
parser.add_argument('-bs', '--batchSize', type=int, default=8)
# parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for the Sigmoid')
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--resolution', type=str, default='32', help='Which resolution to use.')
parser.add_argument('--plot', type=bool,default=True)
parser.add_argument('--loss', type=str, default='iou')
parser.add_argument('-trainsplit', type=float,default=0.25)
# parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('-v', type=bool, default=False)
parser.add_argument('-dataset', type=str, default='brats3dDataset')

# add learning rate
# add valid split
args = parser.parse_args()
dataPath = args.dataPath
#NUM_WORKERS = args.numWorkers
NUM_EPOCHS = args.numEpochs
BATCH_SIZE = args.batchSize
SPLIT_FRAC = args.trainsplit
# LEARNING_RATE = args.lr
# THRESHOLD = args.threshold
RESOLUTION = args.resolution
LOSS = args.loss

print(f"Data path is {dataPath}.")
print(f"NumEpochs is {NUM_EPOCHS}.")
print(f"BatchLength is {BATCH_SIZE}.")
print(f"TrainValid split is {SPLIT_FRAC}.")


# to be set by the parser
VERBOSE = args.v

# Get data
import theDataset
datasetname = args.dataset
fullDataset = theDataset.datasetname(dataPath)

print(f"There are {len(fullDataset)} images in total.")

# Split into training and validation
valid_size = int(SPLIT_FRAC * len(fullDataset))
train_size = len(fullDataset) - valid_size
train_dataset, valid_dataset = random_split(fullDataset, [train_size, valid_size])
print(f"There are {len(train_dataset)} training images, and {len(valid_dataset)} validation images.")

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Use model from themodel.py
device = 'cpu'
if args.cuda and torch.cuda.is_available():
    device = 'cuda'


# model, optimizer, loss function
torchDevice = torch.device(device)
model = SmallU3D().to(torchDevice)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if args.loss == 'dice':
    from themodel import diceLossModule
    criterion = diceLossModule()
else:
    criterion = torch.nn.BCEWithLogitsLoss()


# Here starts the training
print("\nAll right, let's do this.")

epochMeanLosses = []
epochMeanScores = []
for epoch in range(NUM_EPOCHS):
    # print(f'This is epoch number {epoch}.')
    print(f'\n-------Epoch {epoch+1}-------')

    # training loop----
    print('\n*Training')
    model.train()
    losses = []
    batchloop = tqdm.tqdm(train_dataloader)
    scores = []
    for x,y in batchloop:
        # use cuda if available
        x = x.to(torchDevice)
        y = y.to(torchDevice)
        # Forward pass
        y_pred = model(x)
        # Compute loss
        loss = criterion(y_pred,y)
        # Kill gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # update gradients
        optimizer.step()

        batchloop.set_description(f"Epoch number {epoch+1}, Loss: {loss.item()}")
        losses.append(loss.item())

        score = iouscore(y_pred,y)
        scores.append(score)

    print(f"I trained on {len(losses)} images. The average training loss was {np.asarray(losses).mean()}.\n")
    print(f"The average training score was {np.asarray(scores).mean()}.\n")

    # validation loop----
    print('\n*Validation')
    model.eval()
    with torch.no_grad():
        losses = []
        scores = []
        validloop = tqdm.tqdm(valid_dataloader)
        for x,y in validloop:
            x = x.to(torchDevice)
            y = y.to(torchDevice)
            y_pred = model(x)
            loss = criterion(y_pred,y)
            losses.append(loss.item())
            validloop.set_description('Loss: {}'.format(loss.item()))

            score = iouscore(y_pred,y)
            scores.append(score)

        print(f"I evaluated the model on {len(scores)} images")
        
        epochMeanLoss = np.asarray(losses).mean().item()
        print(f"The avg validation loss is {epochMeanLoss}.")
        epochMeanLosses.append(epochMeanLoss)

        epochMeanScore = np.asarray(scores).mean().item()
        print(f"The avg IoU score is: {epochMeanScore}.")
        epochMeanScores.append(epochMeanScore)

        # save/overwrite losses and scores
        np.save('losses', epochMeanLosses)
        np.save('scores',epochMeanScores)

print('\nWhile validating, these were the mean losses:\n')
print(epochMeanLosses)
print('\nWhile validating, these were the mean scores:\n')
print(epochMeanScores)
print("\nI am saving the current model now.")
torch.save(model.state_dict(), 'model.pt')

if args.plot:
    import matplotlib.pyplot as plt
    plt.plot(epochMeanLosses)
    plt.plot(epochMeanScores)
    plt.show()

# To reload it: 
# model = myModel()
# model.load_state_dict(torch.load(PATH))
# model.eval()