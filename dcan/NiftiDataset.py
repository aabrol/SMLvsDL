import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from utils import load_nifti

# MRIDataset should be used rather than this class.
class NiftiDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels.index)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = load_nifti(img_path)
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
