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
parser.add_argument('-dp', type=str, default='/rsrch1/ip/jrcalabrese')
parser.add_argument('-ne', type=int, default=4)
parser.add_argument('-bs', type=int, default=8)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-resolution', type=str, default='32', help='Which resolution to use.')
parser.add_argument('-trainsplit', type=float,default=0.25)
parser.add_argument('-model', type=str, default='ConvSeq')
parser.add_argument('-dataset', type=str, default='brats3dDataset')
parser.add_argument('-loss', type=str, default='iou')
parser.add_argument('-score', type=str, default='iou')
parser.add_argument('-optim', type=str, default='SGD')
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-device', type=str, default='cuda')
parser.add_argument('-plot', type=bool,default=True)
parser.add_argument('-savebest', type=bool, default=True)
parser.add_argument('-makegif', type=bool, default=True)


args = parser.parse_args()

# global variables for later use
NUM_EPOCHS = args.ne
BATCH_SIZE = args.bs
LR = args.lr
MOMENTUM = args.momentum
RESOLUTION = args.resolution
if RESOLUTION == '0':
    dataPath = os.path.join(args.dp, 'numpyDataOG')
else:
    dataPath = os.path.join(args.dp, 'numpyData'+RESOLUTION)
SPLIT_FRAC = args.trainsplit

# start logging
def gettime():
    now = datetime.datetime.now()
    return now.strftime('%H%M%S')

now = datetime.datetime.now()
start_time = now.strftime('%y%m%d_%H%M%S')
logPath = os.path.join('models', start_time + '.log')

def add2log(s,logPath=logPath,display=True):
    with open(logPath, 'a') as f:
        if display:
            print(s)
        f.write(s)
    return None

from extra.name_generator import generate_name
rn = generate_name(fmodelPath='extra/firstnames.pt',lmodelPath='extra/lastnames.pt',vocabPath='extra/lastnames.pkl')
add2log(rn + ' strated training.\n')
add2log(f"Model is {args.model}.\n")
add2log(f"Resolution is {RESOLUTION}.\n")
add2log(f"Loss is {args.loss}.\n")
add2log(f"Score is {args.score}.\n")
add2log(f"NumEpochs is {NUM_EPOCHS}.\n")
add2log(f"Batch Length is {BATCH_SIZE}.\n")
add2log(f"Optimizer is {args.optim}.\n")
add2log(f"Learning rate is {LR}.\n")
add2log(f"Momentum is {MOMENTUM}.\n")
add2log(f"Valid/Train ratio is {SPLIT_FRAC}.\n")
add2log(f"Data path is {dataPath}.\n")

# Get data
import thedataset
add2log(f"\n{gettime()} {rn} is loading data using the {args.dataset} class.\n")
datasetname = args.dataset
datasetClass = getattr(thedataset, datasetname)
fullDataset = datasetClass(dataPath)
add2log(f"\tThere are {len(fullDataset)} images in total.\n")

# Split into training and validation
valid_size = int(SPLIT_FRAC * len(fullDataset))
train_size = len(fullDataset) - valid_size
train_dataset, valid_dataset = random_split(fullDataset, [train_size, valid_size])
add2log(f"\tThere are {len(train_dataset)} training images, and {len(valid_dataset)} validation images.\n")

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
add2log(f'{gettime()} Data is loaded.\n')

# set device
if args.cuda and torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device('cpu')

print(f"\nDevice used is {device}.")

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
add2log(f"\n{gettime()} Model {modelname} was initialized from scratch.\n")

# set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
add2log(f"\tOptimizer SGD with learning rate {LR} and momentum {MOMENTUM}.\n")

# loss function
import thescore
if args.loss == 'iou':
    lossClass = getattr(thescore, args.loss+'Module')
else:
    raise KeyError("Only have iou for now.")
criterion = lossClass()
# criterion = nn.BCEWithLogitsLoss()
add2log(f"\tLoss and Score are IoU.\n")

# score function
from thescore import iou_score
score_fun = iou_score

# Here starts the training
add2log("\nAll right, let's do this.\n")

epochLosses = []
epochScores = []
epochTrainScores = []

best_score = [0.0, -1]

try:
    for epoch in range(NUM_EPOCHS):
        add2log(f'\n-------Epoch {epoch+1}-------\n')
        add2log(f"{gettime()} {rn} started epoch {epoch+1}.\n")

        # training loop----
        add2log('\n\t---Training-------------\n\n')
        model.train()
        trainlosses = []
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
            y_pred = torch.sigmoid(y_pred) # remove if using BCEWithLogits, or similar
            loss = criterion(y_pred,y)
            # Backward pass
            loss.backward()
            # update
            optimizer.step()

            batchloop.set_description(f"Epoch number {epoch+1}, Loss: {loss.item()}")
            trainlosses.append(loss.item())

            y_pred = y_pred.round()
            y_pred = y_pred.byte()
            y = y.byte()
            score = score_fun(y_pred,y)
            trainscores.append(score.item())

        # for loss in trainlosses:
        #     add2log(f"\tBatch loss is {loss}.\n", display=False)
        # add2log("\n")
        # for score in trainscores:
        #     add2log(f"\tBatch score is {score}.\n",display=False)



        avgloss = np.asarray(trainlosses).mean()
        add2log(f"\nThe average training loss was {avgloss}.\n")
        epochLosses.append(avgloss)

        avgTrainScore = np.asarray(trainscores).mean()
        add2log(f"The average training score was {avgTrainScore}.\n")
        epochTrainScores.append(avgTrainScore)

    # -------------

        # validation loop----
        add2log('\n\t---Testing-------------\n\n')
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
                scores.append(score.item())
            
            # for score in scores:
            #     add2log(f"\tBatch score is {score}.\n", display=False)

            avgscore = np.asarray(scores).mean()
            epochScores.append(avgscore)
            add2log(f"\nThe average testing score was {avgscore}.\n")

            if avgscore > best_score[0]:
                add2log('The score improved!\n')
                best_score[0] = avgscore
                best_score[1] = epoch+1
                
                if args.savebest:
                    rootpath = '/rsrch1/ip/jrcalabrese'
                    modelPath = os.path.join(rootpath, 'models', start_time + '_'  + rn + '.pt')
                    add2log(f"Saving model from epoch {epoch+1}.\n{modelPath}\n")
                    torch.save(model.state_dict(), modelPath)                

            # save/overwrite losses and scores
            # np.save(os.path.join('models','losses'), epochLosses)
            # np.save(os.path.join('models','scores'),epochScores)

    add2log(f"\n{gettime()} {rn} is done training.\n")

except KeyboardInterrupt:
    add2log(f'\n{gettime()} Training was interrupted by KeyboardInterrupt.\n')

add2log(f"The last training score was {epochTrainScores[-1]}.\n")
add2log(f"The best model was achieved during epoch {best_score[1]}, with an average validation score of {best_score[0]}.")

# print('While training, these were the mean losses:\n')
# print(epochLosses)
# print('\nWhile training, these were the mean scores:\n')
# print(epochTrainScores)
# print('\nWhile validating, these were the mean scores:\n')
# print(epochScores)

# I already saved the best model.
if not args.savebest:
    rootpath = '/rsrch1/ip/jrcalabrese'
    modelPath = os.path.join(rootpath, 'models', start_time + '_model' + '.pt')
    add2log(f"Saving model.\n{modelPath}")
    torch.save(model.state_dict(), modelPath)

if args.plot:
    import matplotlib.pyplot as plt
    plt.figure()

    plt.subplot(211)
    plt.title(f'{rn} - {modelname}')
    plt.plot(epochLosses,'b', label='Training losses')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochScores,'g', label='Validation scores')
    plt.plot(epochTrainScores,'r', label='Training scores')
    plt.legend()

    plotPath = os.path.join('models', start_time + '_plot' + '.png')
    plt.savefig(plotPath)
    plt.show()

if args.makegif:
    gifmodel = modelClass()
    gifmodel.to(device)
    gifmodel.load_state_dict(torch.load(modelPath))
    x,y = valid_dataset.__getitem__(0)
    x = x.to(device)
    x = x.unsqueeze(0)
    y = y
    gifmodel.eval()
    y_pred = gifmodel(x)
    y_pred = torch.sigmoid(y_pred).round().squeeze(dim=0)

    import matplotlib.pyplot as plt
    r = y.shape[2]
    for i in range(r):
        plt.figure()
        npy = y.numpy()[:,:,i]
        plt.axis('off')
        plt.imshow(npy)
        plt.savefig('ignore/pregif/img'+f"{i:02d}"+'.png')

    import imageio
    images = []
    for i in range(r):
        filename = 'ignore/pregif/img'+f"{i:02d}"+'.png'
        images.append(imageio.imread(filename))
    targetGifPath = os.path.join('models', start_time + '_target'+'.gif')
    imageio.mimsave(targetGifPath, images, duration=.2)

    for i in range(r):
        plt.figure()
        npy_pred = y_pred.cpu().detach().numpy()[:,:,i]
        plt.axis('off')
        plt.imshow(npy_pred)
        plt.savefig('ignore/pregif/img'+f"{i:02d}"+'.png')
    images = []
    for i in range(r):
        filename = 'ignore/pregif/img'+f"{i:02d}"+'.png'
        images.append(imageio.imread(filename))
    predGifPath = os.path.join('models', start_time + '_pred'+'.gif')
    imageio.mimsave(predGifPath, images, duration=.2)
        

"""
To reload models: 
    model = myModel()
    model.load_state_dict(torch.load(PATH))
    model.eval()
NOTE: jupyter might print some 'incompatible keys' nonsense. Just ignore.
"""