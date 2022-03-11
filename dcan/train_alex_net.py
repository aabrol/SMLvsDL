import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcan.NiftiDataset import NiftiDataset

shared_dir = '/home/feczk001/shared'
experiments_dir = os.path.join(shared_dir, 'projects/S1067_Loes/experiments')
images_dir = \
    os.path.join(shared_dir, 'data/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task516_Paper_Fold0')

# Prepare data for training with DataLoaders
training_annotations_file = os.path.join(experiments_dir, 'training_labels.csv')
training_image_dir = os.path.join(images_dir, 'imagesTr')
training_data = NiftiDataset(training_annotations_file, training_image_dir)
validation_loader = DataLoader(training_data, batch_size=64, shuffle=True)

test_annotations_file = os.path.join(experiments_dir, 'training_labels.csv')
test_image_dir = os.path.join(images_dir, 'imagesTs')
test_data = NiftiDataset(test_annotations_file, test_image_dir)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
# Display image and label.
train_features, train_labels = next(iter(validation_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0][0][91].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Get Device for Training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

# Per-Epoch Activity
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.
