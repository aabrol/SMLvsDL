import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from dcan.NiftiDataset import NiftiDataset
from models import AlexNet3D_Dropout

from torch.utils.data import DataLoader

shared_dir = '/home/feczk001/shared'
experiments_dir = os.path.join(shared_dir, 'projects/S1067_Loes/experiments')
images_dir = \
    os.path.join(shared_dir, 'data/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task516_Paper_Fold0')

# TODO Use MRIDataset rather than NiftiDataset once I figure out how MRIDataset works.
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

model = AlexNet3D_Dropout(num_classes=9).to(device)
print(model)

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# The Training Loop
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(validation_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(validation_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Per-Epoch Activity
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
