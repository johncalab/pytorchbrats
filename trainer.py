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
#parser.add_argument('-nw', '--numWorkers', type=int, help='NumWorkers', default=2)
parser.add_argument('-ne', '--numEpochs', type=int, default=3, help='NumEpochs')
parser.add_argument('-bs', '--batchSize', type=int, default=16, help='BatchSize')
# add learning rate
# add valid split
args = parser.parse_args()
dataPath = args.dataPath
#NUM_WORKERS = args.numWorkers
NUM_EPOCHS = args.numEpochs
BATCH_SIZE = args.batchSize
SPLIT_FRAC = 0.25
LEARNING_RATE = 1e-4

print('dataPath =', dataPath)
#print('NumWorkrs =', NUM_WORKERS)
print(f'NumEpochs = {NUM_EPOCHS}')
print(f'BatchSize = {BATCH_SIZE}')
print(f'SPLIT_FRAC = {SPLIT_FRAC}')







# Get data
# dataPath = 'data'
#dataPath = os.path.join('ignore', 'playData')
fullDataset = bratsDataset(dataPath)
print(f"There are {len(fullDataset)} images.")

valid_size = int(SPLIT_FRAC * len(fullDataset))
train_size = len(fullDataset) - valid_size
train_dataset, valid_dataset = random_split(fullDataset, [train_size, valid_size])
print(f"There are {len(train_dataset)} training images, and {len(valid_dataset)} validation images.")

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)#, num_workers=NUM_WORKERS)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)#, num_workers=NUM_WORKERS)

# Use model from themodel.py
device = "cuda" if torch.cuda.is_available() else "cpu"
torchDevice = torch.device(device)
model = SmallU3D().to(torchDevice)
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
    
    batchloop = tqdm.tqdm(train_dataloader)
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

        batchloop.set_description(f"Epoch number {epoch}, Loss: {loss.item()}")
        losses.append(loss.item())

    print(f"I trained on {len(losses)} images. The average loss was {sum(losses)/len(losses)}.")

    # validation ----
    validlosses = []
    model.eval()
    with torch.no_grad():
        validloop = tqdm.tqdm(valid_dataloader)
        scores = []
        for x,y in validloop:
            y_pred = model(x)
            loss = criterion(y,y_pred)
            validlosses.append(loss.item())
            validloop.set_description('Loss: {}'.format(loss.item()))

            ny = y.cpu().numpy()
            ny_pred = y_pred.cpu().numpy()
            intersection = np.logical_and(ny,ny_pred)
            union = np.logical_or(ny,ny_pred)
            iouscore = np.sum(intersection) / np.sum(union)
            scores.append(iouscore)

        print(f"I evaluated the model on {len(scores)} images")
        print("The avg validation loss is {}".format(sum(validlosses)/len(validlosses)))
        print(f"The avg IoU score is: {np.asarray(scores).mean()}")

print("I am saving the current model now.")
torch.save(model.state_dict(), 'model.pt')
# To reload it: 
# model = myModel()
# model.load_state_dict(torch.load(PATH))
# model.eval()
