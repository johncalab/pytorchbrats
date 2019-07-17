# Pytorch BraTS
Learning a bit of PyTorch by experimenting with medical data. http://medicaldecathlon.com/  
(BraTS stands for Brain Tumor Segmentation)

The repository consists of the following files:
* `n1`, `n2`, `n3`: expository jupyter notebooks,
* `convert_to_np.py`: a script to convert and downsample input data,
* `trainer.py`: controls the main workflow to train a model,
* `thedataset.py`: contains the pytorch Dataset class to feed images to the model,
* `thescore.py`: contains code for the IoU score and loss functions,
* `themodel.py`: contains code for the various models used.

* `models` is a folder containing models (as `.pt` files, when not too big), logs, plots, and gifs comparing predictions with targets,
* `extra` contains auxiliary files, mostly for the expository notebooks, with the exception of
* `name_generator.py` which is used to generate 'human' names for the models, and which in turn relies on `firstnames.pt`, `lastnames.pt` and `lastnames.pkl`. The first two are (crudely pretrained) pytorch models, while the latter is a dictionary to decode the output of the model.

## Models

Input images are 4d nifti files, output images are 3d nifti files.
More details may be found in [`n1-exploration.ipynb`](n1-exploration.ipynb). 
The models treat input images as 4D (3d images + 4 different scan modalities), and output images as 3d.
`themodel.py` contains the following models:
* Crush - a naive two-layer linear model which brutally compresses images,
* ConvSeq - four (3d) CNN layers,
* Small3dUcat - a mini U-net which downsamples only once, skip connections are via concatenating layers,
* Small3dUadd - same except skip connections are via addition of tensors,
* Unet3d - the classic U-net from the [original paper](https://arxiv.org/abs/1505.04597), possibly with a difference in activations,
* UU3d - a variant of the U-net proposed in [this paper](https://arxiv.org/pdf/1701.03056.pdf).

Score and loss are computed via "Intersection over Union" or Jaccard index.
More details may be found in [`n2-models.ipynb`](n2-models.ipynb).

## Trainer
The script `trainer.py` contains code to train a model.
* Arguments may be passed via the parser to choose model, number of epochs, etc.
* The data is first loaded to a pytorch DataLoader.
* A model is initialized and training begins.
* A 'human' name is assigned to a model via a character-level RNN previously trained on a list of about a thousand given names.
* A log file is created, and average scores and losses per epoch are recorded.
* Each time the score increases on the validation set, the model is saved (overwriting the previous one).
* At the end a plot is made of all scores and losses and saved to file.
* A gif comparing slice-by-slice the prediction of the model with target corresponding to a fixed image (`__getitem__(0)`).

The notebook [`n3-postoperative.ipynb`](n3-postoperative.ipynb) summarizes the results of a few trained models.

## Desiderata
A few of the infinitely many things missing:
* normalizing data before training (although batchnorm should compensate for that),
* data augmentation,
* treating slices as 2d images and training 2d models for those,
* a variant of the above, but using different orthogonal projections,
* experimenting with different training methods (optimizer, learning rate, scheduler, etc.),
* training on different resolutions,
* training same model multiple times to compare results,
* adding dropout (and overall tinkering with activation layers).
