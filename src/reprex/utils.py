import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import nibabel as nib
from models import AlexNet3D_Dropout
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, \
    mean_squared_error, r2_score
from dataclasses import dataclass
from scipy.stats import pearsonr


@dataclass
class Config:
    iter: int = 0  # slurmTaskIDMapper maps this variable using tr_smp_sizes and nReps to tss and rep
    tr_smp_sizes: tuple = (100, 200, 500, 1000, 2000, 5000, 10000)
    nReps: int = 20
    nc: int = 10
    bs: int = 16
    lr: float = 0.001
    es: int = 1
    pp: int = 1
    es_va: int = 1
    es_pat: int = 40
    ml: str = '../../temper/'
    mt: str = 'AlexNet3D_Dropout'
    ssd: str = '../../SampleSplits/'
    scorename: str = 'label'
    cuda_avl: bool = True
    nw: int = 8
    cr: str = 'clx'
    tss: int = 100  # modification automated via slurmTaskIDMapper
    rep: int = 0  # modification automated via slurmTaskIDMapper


class MRIDataset(Dataset):

    def __init__(self, cfg, mode):
        self.df = readFrames(cfg.ssd, mode, cfg.tss, cfg.rep)
        self.scorename = cfg.scorename
        self.cr = cfg.cr

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X, y = read_X_y_5D_idx(self.df, idx, self.scorename, self.cr)
        return [X, y]


def readFrames(ssd, mode, tss, rep):
    # Read Data Frame
    file_path = os.path.join(ssd, mode + '_' + str(tss) + '_rep_' + str(rep) + '.csv')
    df = pd.read_csv(file_path)

    print('Mode ' + mode + ' :' + 'Size : ' +
          str(df.shape) + ' : DataFrames Read ...')

    return df


def read_X_y_5D_idx(df, idx, scorename, cr):

    X, y = [], []

    # Read image
    fN = df['smriPath'].iloc[idx]
    X = np.float32(nib.load(fN).get_fdata())
    X = (X - X.min()) / (X.max() - X.min())
    X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))

    # Read label
    y = df[scorename].iloc[idx]
    if scorename == 'label':
        y -= 1

    if cr == 'reg':
        y = np.array(np.float32(y))
    elif cr == 'clx':
        y = np.array(y)

    return X, y


def train(dataloader, net, optimizer, criterion, cuda_avl):

    net.train()

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        # Fetch the inputs
        inputs, labels = data

        # Wrap in variable and load batch to gpu
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs[0].squeeze(), labels)
        loss.backward()
        optimizer.step()

    return loss


def test(dataloader, net, cuda_avl, cr):

    net.eval()
    y_pred = np.array([])
    y_true = np.array([])

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        inputs, labels = data

        # Wrap in variable and load batch to gpu
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass
        outputs = net(inputs)

        if cr == 'clx':
            _, predicted = torch.max(outputs[0].data, 1)
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
        elif cr == 'reg':
            y_pred = np.concatenate((y_pred, outputs[0].data.cpu().numpy().squeeze()))

        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))

    return y_true, y_pred


def evalMetrics(dataloader, net, cfg):

    # Batch Dataloader
    y_true, y_pred = test(dataloader, net, cfg.cuda_avl, cfg.cr)

    if cfg.cr == 'clx':

        # Evaluate classification performance
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        return acc, bal_acc

    elif cfg.cr == 'reg':

        # Evaluate regression performance
        mae = mean_absolute_error(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        r, p = pearsonr(y_true, y_pred)

        return mae, ev, mse, r2, r, p

    else:
        print('Check cr flag')


def generate_validation_model(cfg):

    # Initialize net based on model type (mt, nc)
    net = initializeNet(cfg)

    # Training parameters
    epochs_no_improve = 0
    valid_acc = 0

    criterion = None
    if cfg.cr == 'clx':
        criterion = nn.CrossEntropyLoss()
        reduce_on = 'max'
        m_val_acc = 0
        history = pd.DataFrame(columns=['scorename', 'iter', 'epoch',
                                        'tr_acc', 'bal_tr_acc', 'val_acc', 'bal_val_acc', 'loss'])
    elif cfg.cr == 'reg':
        criterion = nn.MSELoss()
        reduce_on = 'min'
        m_val_acc = 100
        history = pd.DataFrame(columns=['scorename', 'iter', 'epoch', 'tr_mae', 'tr_ev', 'tr_mse',
                                        'tr_r2', 'tr_r', 'tr_p', 'val_mae', 'val_ev', 'val_mse', 'val_r2', 'val_r',
                                        'val_p', 'loss'])
    else:
        print('Check config flag cr')

    # Load model to gpu
    if cfg.cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Declare optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)

    # Declare learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode=reduce_on, factor=0.5, patience=7, verbose=True)

    # Batch Dataloader
    trainloader = loadData(cfg, 'tr')
    validloader = loadData(cfg, 'va')

    for epoch in range(cfg.es):

        # Train
        print('Training: ')
        loss = train(trainloader, net, optimizer, criterion, cfg.cuda_avl)
        loss = loss.data.cpu().numpy()

        if cfg.cr == 'clx':

            print('Validating: ')

            # Evaluate classification perfromance on training and validation data
            train_acc, bal_train_acc = evalMetrics(trainloader, net, cfg)
            valid_acc, bal_valid_acc = evalMetrics(validloader, net, cfg)

            # Log Performance
            history.loc[epoch] = [cfg.scorename, cfg.iter, epoch, train_acc,
                                  bal_train_acc, valid_acc, bal_valid_acc, loss]

            # Check for maxima (e.g. accuracy for classification)
            isBest = valid_acc > m_val_acc

        elif cfg.cr == 'reg':

            print('Validating: ')

            # Evaluate regression perfromance on training and validation data
            train_mae, train_ev, train_mse, train_r2, train_r, train_p = evalMetrics(
                trainloader, net, cfg)
            valid_acc, valid_ev, valid_mse, valid_r2, valid_r, valid_p = evalMetrics(
                validloader, net, cfg)

            # Log Performance
            history.loc[epoch] = [cfg.scorename, cfg.iter, epoch, train_mae, train_ev, train_mse, train_r2,
                                  train_r, train_p, valid_acc, valid_ev, valid_mse, valid_r2, valid_r, valid_p, loss]

            # Check for minima (e.g. mae for regression)
            isBest = valid_acc < m_val_acc

        else:
            print('Check cr flag')

        # Write Log
        history.to_csv(cfg.ml + 'history.csv', index=False)

        # Early Stopping
        if cfg.es_va:

            # If minima/maxima
            if isBest:

                # Save the model
                torch.save(net.state_dict(), open(
                    cfg.ml + 'model_state_dict.pt', 'wb'))

                # Reset counter for patience
                epochs_no_improve = 0
                m_val_acc = valid_acc

            else:

                # Update counter for patience
                epochs_no_improve += 1

                # Check early stopping condition
                if epochs_no_improve == cfg.es_pat:

                    print('Early stopping!')

                    # Stop training: Return to main
                    return history, m_val_acc

        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_acc)


def evaluate_test_accuracy(cfg):

    # Load validated net
    net = loadNet(cfg)
    net.eval()

    # Dataloader
    testloader = loadData(cfg, 'te')

    if cfg.cr == 'clx':

        # Initialize Log File
        outs = pd.DataFrame(columns=['iter', 'acc_te', 'bal_acc_te'])

        print('Testing: ')

        # Evaluate classification performance
        acc, bal_acc = evalMetrics(testloader, net, cfg)

        # Log Performance
        outs.loc[0] = [cfg.iter, acc, bal_acc]

    elif cfg.cr == 'reg':

        # Initialize Log File
        outs = pd.DataFrame(columns=[
                            'iter', 'mae_te', 'ev_te', 'mse_te', 'r2_te', 'r_te', 'p_te'])

        print('Testing: ')

        # Evaluate regression performance
        mae, ev, mse, r2, r, p = evalMetrics(testloader, net, cfg)

        # Log Performance
        outs.loc[0] = [cfg.iter, mae, ev, mse, r2, r, p]

    else:
        print('Check cr mode')

    # Write Log
    outs.to_csv(cfg.ml+'test.csv', index=False)


def loadData(cfg, mode):

    # Batch Dataloader
    # doesn't seem to be working; tried 1, 2, 4, 8, 16, 32 - mem used stays the same! need to verify the MRIdataset
    # custom functionality maybe
    prefetch_factor = 8
    dset = MRIDataset(cfg, mode)

    dloader = DataLoader(dset, batch_size=cfg.bs,
                         shuffle=True, num_workers=cfg.nw, drop_last=True, pin_memory=True,
                         prefetch_factor=prefetch_factor, persistent_workers=True)

    return dloader


def loadNet(cfg):

    # Load validated model
    net = initializeNet(cfg)
    model = torch.nn.DataParallel(net)
    net = 0
    net = load_net_weights2(model, cfg.ml+'model_state_dict.pt')

    return net


def updateIterML(cfg):

    # Update Iter (in case of multitask training)
    if cfg.pp:
        cfg.iter += 1

    # Map slurmTaskID to training sample size (tss) and CV rep (rep)
    cfg = slurmTaskIDMapper(cfg)

    # Update Model Location
    cfg.ml = cfg.ml+cfg.mt+'_scorename_'+cfg.scorename+'_iter_' + \
        str(cfg.iter)+'_tss_'+str(cfg.tss)+'_rep_'+str(cfg.rep)+'_bs_'+str(cfg.bs)+'_lr_' + \
        str(cfg.lr)+'_espat_'+str(cfg.es_pat)+'/'

    # Make Model Directory
    if not os.path.isdir(cfg.ml):
        os.makedirs(cfg.ml)

    return cfg


def slurmTaskIDMapper(cfg):

    # Map iter value (slurm taskID) to training sample size (tss) and crossvalidation repetition (rep)
    tv, rv = np.meshgrid(cfg.tr_smp_sizes, np.arange(cfg.nReps))
    tv = tv.reshape((1, np.prod(tv.shape)))
    rv = rv.reshape((1, np.prod(tv.shape)))
    tss = tv[0][cfg.iter]
    rep = rv[0][cfg.iter]
    print(tss, rep)
    cfg.tss = tss
    cfg.rep = rep
    print(cfg.iter, cfg.tss, cfg.rep)

    return cfg


def initializeNet(cfg):

    # Initialize net based on model type (mt, nc)
    net = None
    if cfg.mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=cfg.nc)
    else:
        print('Check model type')

    return net


def load_net_weights2(net, weights_filename):

    # Load trained model
    state_dict = torch.load(
        weights_filename,  map_location=lambda storage, loc: storage)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state)

    return net
