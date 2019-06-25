import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from thedataset import bratsDataset
from themodel import SmallU3D
import os
import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import tqdm
import numpy as np

# parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataPath', type=str, help="Use -d 'dataPath' to specify location of data. Default is 'data'.", default='data')
parser.add_argument('-nw', '--numWorkers', type=int, help='NumWorkers', default=2)
parser.add_argument('-ne', '--numEpochs', type=int, default=3, help='NumEpochs')
parser.add_argument('-bs', '--batchSize', type=int, default=16, help='BatchSize')
# add learning rate
# add valid split
args = parser.parse_args()
dataPath = args.dataPath
NUM_WORKERS = args.numWorkers
NUM_EPOCHS = args.numEpochs
BATCH_SIZE = args.batchSize
SPLIT_FRAC = 0.25
LEARNING_RATE = 1e-4

print('dataPath =', dataPath)
print('NumWorkrs =', NUM_WORKERS)
print(f'NumEpochs = {NUM_EPOCHS}')
print(f'BatchSize = {BATCH_SIZE}')
print(f'SPLIT_FRAC = {SPLIT_FRAC}')







# Get data
# dataPath = 'data'
#dataPath = os.path.join('ignore', 'playData')
fullDataset = bratsDataset(dataPath)

valid_size = int(SPLIT_FRAC * len(fullDataset))
train_size = len(fullDataset) - valid_size
train_dataset, valid_dataset = random_split(fullDataset, [train_size, valid_size])

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Use model from themodel.py
model = SmallU3D()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
from themodel import diceLossModule
criterion = diceLossModule()

# Here starts the training
print("All right, I am starting the training.")
for epoch in range(NUM_EPOCHS):
    print(f'This is epoch number {epoch}.')

    # training ----
    model.train()

    losses = []
    
    description = 'Epoch number ' + str(epoch+1)
    batchloop = tqdm.tqdm(train_dataloader, desc=description)

    for x,y in batchloop:

        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = criterion(y,y_pred)

        # Kill gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # update gradients
        optimizer.step()

        batchloop.set_description("Loss: {}".format(loss.item()))
        losses.append(loss.item())

    print('Epoch number {} of {}. Avg batchloss is {}'.format(str(epoch+1),NUM_EPOCHS,sum(losses)/len(losses)))

    # validation ----
    validlosses = []
    model.eval()
    with torch.no_grad():
        validloop = tqdm.tqdm(valid_dataloader)
        for x,y in validloop:

            # here I would write x.to(device) to use gpu
            y_pred = model(x)
            loss = criterion(y,y_pred)

            validlosses.append(loss.item())

            validloop.set_description('Loss: {}'.format(loss.item()))

        print('Avg validation loss is {}'.format(sum(validlosses)/len(validlosses)))

print("I am saving the current model now.")
torch.save(model.state_dict(), 'model.pt')
# To reload it: 
# model = myModel()
# model.load_state_dict(torch.load(PATH))
# model.eval()
