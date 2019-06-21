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

# if parser is added, these variables should become options
SPLIT_FRAC = 0.25
BATCH_SIZE = 3
NUM_WORKERS = 2
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4

# Get data
# dataPath = 'data'
dataPath = os.path.join('ignore', 'playData')
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
for epoch in range(NUM_EPOCHS):

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

torch.save(model.state_dict(), 'model.pt')
# To reload it: 
# model = myModel()
# model.load_state_dict(torch.load(PATH))
# model.eval()
