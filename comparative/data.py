import numpy as np
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import pandas as pd


class MRIDataset(Dataset):

    def __init__(self, tss, rep, mode, ssd, scorename, regression):
        self.df = readFrames(tss, rep, mode, ssd)
        self.scorename = scorename
        self.regression = regression

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X, y = read_X_y_5D_idx(self.df, idx, self.scorename, self.regression)
        return [X, y]


def readFrames(tss, rep, mode, ssd):

    if mode == 'tr':
        df = pd.read_csv(ssd + 'tr_' + str(tss) + '_rep_' + str(rep) + '.csv')
    elif mode == 'va':
        df = pd.read_csv(ssd + 'va_' + str(tss) + '_rep_' + str(rep) + '.csv')
    elif mode == 'te':
        df = pd.read_csv(ssd + 'te_' + str(tss) + '_rep_' + str(rep) + '.csv')
    else:
        print('Pick a Valid Mode: tr, va, te')

    print('Rep ' + str(rep) + ' Mode ' + mode + ' :' + 'Size : ' +
          str(df.shape) + ' : DataFrames Read ...')
    return df


def read_X_y_5D_idx(df, idx, scorename, regression):
    X, y = [], []
    fN = df['smriPath'].iloc[idx]
    la = df[scorename].iloc[idx]
    if scorename == 'label':
        la -= 1
    im = np.float32(nib.load(fN).get_fdata())
    im = (im - im.min()) / (im.max() - im.min())
    im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
    X = np.array(im)
    if regression:
        y = np.array(np.float32(la))
    else:
        y = np.array(la)
    return X, y


def get_dataloaders(tss, rep, ssd, num_workers, batch_size=4, scorename='age'):

    if scorename == 'label':  # AgeSexC
        regression = False
        output_dim = 10
    elif scorename == 'sex':  # SexC
        regression = False
        output_dim = 2
    elif scorename == 'age':  # AgeR
        regression = True
        output_dim = 1
    else:
        raise ValueError

    print('loading data rep ' + str(rep))

    trainset = MRIDataset(tss, rep, 'tr', ssd, scorename, regression)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    validset = MRIDataset(tss, rep, 'va', ssd, scorename, regression)
    validloader = DataLoader(validset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    testset = MRIDataset(tss, rep, 'te', ssd, scorename, regression)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    return {'train': trainloader, 'test': validloader, 'val': testloader}, regression, output_dim
