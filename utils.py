import numpy as np
import pandas as pd
import nipy
import scipy as sp
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC
from sklearn.random_projection import GaussianRandomProjection
from hypopt import GridSearch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from models import AlexNet3D_Dropout, AlexNet3D_Deeper_Dropout, AlexNet3D_Dropout_Regression
import numpy as np
import pandas as pd
import nipy
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score 
import time
import sys
import threading
import queue
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy import ndimage

def read_X_y(df, mask, scorename):
    X,y = [],[]
    mask = np.float32(nipy.load_image(mask).get_data()).flatten()
    mask = mask != 0
    for sN in np.arange(df.shape[0]):
        fN = df['smriPath'].iloc[sN]
        la = df[scorename].iloc[sN] 
        if scorename == 'label':
            la -= 1
        im = np.float32(nipy.load_image(fN).get_data())
        im = im.flatten()
        im = im[mask]
        im = (im - im.min()) / (im.max() - im.min())
        X.append(im)
        y.append(la)
    X = np.array(X)
    y = np.array(y)
    return X, y
    
def red_dim(X_tr, y_tr, X_va, X_te, nFeats, meth):

    if meth == 'UFS':
       # 1. UFS
        ufs = SelectKBest(score_func=f_classif, k=nFeats)
        X_tr = ufs.fit_transform(X_tr, y_tr)
        X_va = ufs.transform(X_va)
        X_te = ufs.transform(X_te)
    elif meth == 'RFE':
        # 2. RFE
        rfe = RFE(SVC(kernel="linear", C=1), nFeats, step=0.25)
        rfe = rfe.fit(X_tr, y_tr)
        X_tr = X_tr[:,rfe.support_]
        X_va = X_va[:,rfe.support_]
        X_te = X_te[:,rfe.support_]
    elif meth == 'GRP':
        # 3. GRP
        grp = GaussianRandomProjection(n_components=nFeats)
        X_tr = grp.fit_transform(X_tr, y_tr)
        X_va = grp.transform(X_va)
        X_te = grp.transform(X_te)
    else:
        print('Check Dim. Red. Method')

    print(meth, ': ',  X_tr.shape, X_va.shape, X_te.shape)
    
    return X_tr, X_va, X_te

def readFeaturesLabels(task,tss,rep,drm):    
    """
    # fdir: str, Features directory [100, 200, 500, 1000, 2000, 5000, 10000]
    # sdir: str, Dataframe splits directory
    # task: str, Classification Task
    # tss: int, Sample Size
    # rep: int, Replicate Number (0 to 19)
    # drm: str, Dimensionality Reduction Method (UFS or RFE or GRP)
    # x_tr: np.array, training features
    # x_va: np.array, validation features
    # x_te: np.array, testing features
    # y_tr: np.array, training features
    # y_va: np.array, validation labels
    # y_te: np.array, testing labels
    """
    if task == 'AgeSex':
        fdir = './sMRI_feats/'
        sdir = './SampleSplits/'
        score = 'label'
    elif task == 'Age' or task == 'AgeReg':    
        fdir = './sMRI_feats_Age/'
        sdir = './SampleSplits_Age/'
        score = 'age'
    elif task == 'Sex':    
        fdir = './sMRI_feats_Sex/'
        sdir = './SampleSplits_Sex/'
        score = 'sex'
    elif task == 'REG_MMSE':    
        fdir = './sMRI_feats_MMSE/'
        sdir = './SampleSplits_MMSE/'
        score = 'MMSE'
    else:
        print('Verify task and split dirs..')

    x_tr = np.float32(pd.read_csv(fdir+'X_tr_'+drm+str(tss)+'_rep_'+str(rep)+'.csv',header=None))
    x_va = np.float32(pd.read_csv(fdir+'X_va_'+drm+str(tss)+'_rep_'+str(rep)+'.csv',header=None))
    x_te = np.float32(pd.read_csv(fdir+'X_te_'+drm+str(tss)+'_rep_'+str(rep)+'.csv',header=None))
    
    df_tr = pd.read_csv(sdir+'tr_'+str(tss)+'_rep_'+str(rep)+'.csv')
    df_va = pd.read_csv(sdir+'va_'+str(tss)+'_rep_'+str(rep)+'.csv')
    df_te = pd.read_csv(sdir+'te_'+str(tss)+'_rep_'+str(rep)+'.csv')
    y_tr = np.int32(df_tr[score])
    y_va = np.int32(df_va[score])
    y_te = np.int32(df_te[score])

    # labels saved as 1-10; transform to 0 to 9
    # sex already stored as 0-1 while  age is continuous 
    if score == 'label':
        y_tr -= 1
        y_va -= 1
        y_te -= 1
    
    return x_tr,x_va,x_te,y_tr,y_va,y_te

def run_SML_Classifiers(meth, x_tr, y_tr, x_va, y_va, x_te, y_te, trd, pp=0, mi=1000):
      
    # postprocess scaling:
    if pp == 1: # explore and use better results
        ss = StandardScaler().fit(np.concatenate((x_tr, x_va)))
        x_tr = ss.transform(x_tr)
        x_va = ss.transform(x_va)
        x_te = ss.transform(x_te)

    nt = 8
    # Hyperparameter grids
    C_range_lin = np.logspace(-20, 10, 10, base=2)
    C_range_ker = np.logspace(-10, 20, 10, base=2)
    Y_range = np.logspace(-25, 5, 10, base=2)    
    coef0Vals = [-1,0,1] # Coefficients for Poly and Sigmoid Kernel SVMs

    param_grid_lr = [{'C': C_range_lin}]
    param_grid_svml = [{'C': C_range_lin, 'gamma': Y_range}]
    param_grid_svmk = [{'C': C_range_ker, 'gamma': Y_range, 'coef0': coef0Vals}]

    if meth == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif meth == 'LR':
        gs = GridSearch(model = LogisticRegression(), param_grid = param_grid_lr, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = LogisticRegression(C = gs.best_params['C'])
    elif meth == 'SVML':
        gs = GridSearch(model = SVC(kernel="linear", max_iter=mi), param_grid = param_grid_svml, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="linear", C = gs.best_params['C'], gamma = gs.best_params['gamma'], max_iter=mi)
    elif meth == 'SVMR':
        gs = GridSearch(model = SVC(kernel="rbf", max_iter=mi), param_grid = param_grid_svmk, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="rbf", C = gs.best_params['C'], gamma = gs.best_params['gamma'], max_iter=mi)
    elif meth == 'SVMP':
        gs = GridSearch(model = SVC(kernel="poly", degree =2, max_iter=mi), param_grid = param_grid_svmk, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="poly", C = gs.best_params['C'], gamma = gs.best_params['gamma'], coef0 = gs.best_params['coef0'], max_iter=mi)
    elif meth == 'SVMS':                
        gs = GridSearch(model = SVC(kernel="sigmoid", max_iter=mi), param_grid = param_grid_svmk, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="sigmoid", C = gs.best_params['C'], gamma = gs.best_params['gamma'], coef0 = gs.best_params['coef0'], max_iter=mi)
    else:
        print('Check Valid Classifier Names')                  

    if trd == 'tr': # correct; use this only
        clf.fit(x_tr, y_tr)
    elif trd == 'tr_val':
        clf.fit(np.concatenate((x_tr, x_va)), np.concatenate((y_tr, y_va)))
    else:
        print('Choose trd as tr or tr_val')
        
    scr = clf.score(x_te, y_te)
    
    return scr

class MRIDataset(Dataset):
    """
    iter_: 
    tr_smp_sizes: 
    nReps: 
    mode:
    ssd: 
    scorename: 
    """
    def __init__(self, iter_, tr_smp_sizes, nReps, mode, ssd, scorename):
        self.df = readFrames(iter_, tr_smp_sizes, nReps, mode, ssd)
        self.scorename = scorename

    def __len__(self):        
        return self.df.shape[0]

    def __getitem__(self, idx):
        X,y = read_X_y_5D_idx(self.df, idx, self.scorename)
        return [X,y] 

def readFrames(iter_,tr_smp_sizes,nReps,mode,ssd):
    """
    Map job id to rep id and sample size
    and read dataframe of subject attributes.   
    iter_: int, Job ID or iteration number
    tr_smp_sizes: list of training sizes (int)
    nReps: int, crossvalidation repetition number
    mode: str, training/validation/testing
    ssd: str, sample splits (partitions) directory path
    df: pd.DataFrame, output dataframe
    """
    tv, rv = np.meshgrid(tr_smp_sizes, np.arange(nReps))
    tv = tv.reshape((1, np.prod(tv.shape)))
    rv = rv.reshape((1, np.prod(tv.shape)))
    #iter_ = int( os.environ['SLURM_ARRAY_TASK_ID'] )
    tss = tv[0][iter_]
    rep = rv[0][iter_]
    print(iter_,tss,rep)

    if mode == 'tr':
        df = pd.read_csv(ssd + 'tr_' + str(tss) + '_rep_' + str(rep) + '.csv')
    elif mode == 'va':    
        df = pd.read_csv(ssd + 'va_' + str(tss) + '_rep_' + str(rep) + '.csv')
    elif mode == 'te':
        df = pd.read_csv(ssd + 'te_' + str(tss) + '_rep_' + str(rep) + '.csv')
    else:
        print('Pick a Valid Mode: tr, va, te')

    print('Mode ' + mode + ' :' + 'Size : ' + str(df.shape)+ ' : DataFrames Read ...')            
    return df

def read_X_y_5D_idx(df, idx, scorename):    
    X,y = [],[]
    fN = df['smriPath'].iloc[idx]
    la = df[scorename].iloc[idx]
    if scorename == 'label':
        la -= 1
    im = np.float32(nipy.load_image(fN).get_data())        
    im = (im - im.min()) / (im.max() - im.min())
    im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
    X = np.array(im)
    y = np.array(la)
    return X, y

def read_X_y_5D(df,scorename):
    X,y = [],[]
    for sN in np.arange(df.shape[0]):
        fN = df['smriPath'].iloc[sN]
        la = df[scorename].iloc[sN]
        if scorename == 'label':
            la -= 1
        im = np.float32(nipy.load_image(fN).get_data())        
        im = (im - im.min()) / (im.max() - im.min())
        im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
        X.append(im)
        y.append(la)
    X = np.array(X)
    y = np.array(y)
    print('X: ',X.shape,' y: ',y.shape)    
    return X, y

def prefetch_map(func, input_iter, prefetch=1, check_interval=5):
    """
    Map a function (func) on a iterable (input_iter), but
    prefetch input values and map them asyncronously as output
    values are consumed.
    prefetch: int, the number of values to prefetch asyncronously
    check_interval: int, the number of seconds to block when waiting
                    for output values.
    """
    result_q = queue.Queue(prefetch)
    error_q = queue.Queue(1)
    done_event = threading.Event()

    mapper_thread = threading.Thread(target=_mapper_loop, args=(func, input_iter, result_q, error_q, done_event))
    mapper_thread.daemon = True
    mapper_thread.start()

    while not (done_event.is_set() and result_q.empty()):
        try:
            result = result_q.get(timeout=check_interval)
        except queue.Empty:
            continue

        yield result

    if error_q.full():
        raise error_q.get()[1]


def _mapper_loop(func, input_iter, result_q, error_q, done_event):
    try:
        for x in input_iter:
            result = func(x)
            result_q.put(result)
    except BaseException:
        error_q.put(sys.exc_info())
    finally:
        done_event.set()


def to_gpu(batch):
    inputs = Variable(batch[0].cuda(non_blocking=True))
    labels = Variable(batch[1].cuda(non_blocking=True))
    return inputs, labels


def train(dataloader, net, optimizer, criterion, cuda_avl, num_prefetch=16):
    if cuda_avl:
        dataloader = prefetch_map(to_gpu, dataloader, prefetch=num_prefetch)
    net.train()
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data        
        # wrap them in Variable
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()        
    return loss


def test(dataloader, net, cuda_avl):
    net.eval()
    y_pred = np.array([])
    y_true = np.array([])
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)       
        outputs = net(inputs)
        _, predicted = torch.max(outputs[0].data, 1)
        y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))
    return y_true, y_pred

def test_reg(dataloader, net, cuda_avl):
    net.eval()
    y_pred = np.array([])
    y_true = np.array([])
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)       
        outputs = net(inputs)
        y_pred = np.concatenate((y_pred, outputs[0].data.cpu().numpy()))
        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))
    return y_true, y_pred

def generate_validation_model(iter_,tr_smp_sizes,nReps,ssd,bs,nw,cuda_avl,mt,lr,ml,es,es_pat,es_va,nc,scorename):                           
    
    if mt == 'AlexNet3D':
        net = AlexNet3D(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=nc)
    elif mt == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=nc)
    else:
        print('Check model type')

    criterion = nn.CrossEntropyLoss()
    
    if mt == 'AlexNet3D_Dropout_Regression':
        criterion = nn.MSELoss()
        print(criterion)
    
    trainset = MRIDataset(iter_, tr_smp_sizes, nReps, 'tr', ssd, scorename)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)

    validset = MRIDataset(iter_, tr_smp_sizes, nReps, 'va', ssd, scorename)
    validloader = DataLoader(validset, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)

    if cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)

    # Early stopping details
    max_val_acc = 0
    epochs_no_improve = 0
    valid_acc = 0

    history = pd.DataFrame(columns=['scorename','iter','epoch','tr_acc','bal_tr_acc','val_acc','bal_val_acc','loss'])
    
    for epoch in range(es):

        # Train
        print('Training:')
        loss = train(trainloader, net, optimizer, criterion, cuda_avl)
        loss = loss.data.cpu().numpy()
        y_true, y_pred = test(trainloader, net, cuda_avl)
        train_acc = accuracy_score(y_true, y_pred)
        bal_train_acc = balanced_accuracy_score(y_true, y_pred)

        # Validate
        print('Validating')
        prev_acc = valid_acc
        y_true, y_pred = test(validloader, net, cuda_avl)
        valid_acc = accuracy_score(y_true, y_pred)
        bal_valid_acc = balanced_accuracy_score(y_true, y_pred)
        history.loc[epoch] = [scorename, iter_, epoch, train_acc, bal_train_acc, valid_acc, bal_valid_acc, loss]
        history.to_csv(ml + 'history.csv', index=False)

        print ('scorename' + scorename + '_Iter '+str(iter_)+' Epoch '+str(epoch)+' Tr. Acc.: '+ str(train_acc)+' Bal. Tr. Acc.: '+ str(bal_train_acc)+' Val. Acc.: '+str(valid_acc)+' Bal. Val. Acc.: '+str(bal_valid_acc)+' Tr. Loss '+str(loss))

        if es_va:  
            # If the validation accuracy is at a maximum
            if valid_acc > max_val_acc:
                # Save the model
                torch.save(net.state_dict(), open(ml + 'model_state_dict.pt', 'wb'))
                epochs_no_improve = 0
                max_val_acc = valid_acc
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == es_pat:
                    print('Early stopping!')                    
                    return history, max_val_acc          
        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_acc)

def load_net_weights2(net, weights_filename):
    state_dict = torch.load(weights_filename,  map_location=lambda storage, loc: storage)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state)
    return net

def forward_pass(X_te,net):
    # used to overcome memory constraints
    net.eval()
    outs30=[]
    ims = X_te.shape
    for n in range(0,ims[0]):    
        im = X_te[n].reshape(1,1,ims[2],ims[3],ims[4])          
        temp = net(im)
        temp0 = temp[0]
        aa = nn.functional.softmax(temp0,dim=1).data.cpu().numpy().squeeze()
        outs30.append(aa)        
    probs = np.vstack((outs30))
    return probs

def evaluate_test_accuracy(iter_,tr_smp_sizes,nReps,mode,ssd,ml,mt,nc,t0,scorename):

    # Load validated model
    net=0    

    if mt == 'AlexNet3D':
        net = AlexNet3D(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=nc)
    elif mt == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=nc)
    else:
        print('Check model type')

    model = torch.nn.DataParallel(net)
    net = 0
    net = load_net_weights2(model, ml+'model_state_dict.pt')

    df_te = readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
    X_te, y_te = read_X_y_5D(df_te,scorename)     
    ims = X_te.shape
    X_te = Variable(torch.from_numpy(X_te))

    # save accuracy as csv file
    outs = pd.DataFrame(columns=['iter_','acc_te','Time_Clx'])    
    probs = forward_pass(X_te,net)
    
    p_y_te = probs.argmax(1)
    acc_te = accuracy_score(y_te,p_y_te)

    # write test accuracy and clx time
    t1 = time.time()
    time_clx = t1 - t0

    outs.loc[0] = [iter_, acc_te, time_clx]
    outs.to_csv(ml+'test_es.csv', index=False)

    return outs, iter_, acc_te

def run_SML_Regressors(meth, x_tr, y_tr, x_va, y_va, x_te, y_te, trd, pp=0):
    
    # postprocess scaling:
    if pp == 1:
        ss = StandardScaler().fit(np.concatenate((x_tr, x_va)))
        x_tr = ss.transform(x_tr)
        x_va = ss.transform(x_va)
        x_te = ss.transform(x_te)

    nt = 8
    # Hyperparameter grids
    param_grid_kr={"kernel": ['linear','rbf','poly','sigmoid'],
                                  "alpha": [1e0, 1e-1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-4, 2, 20)}
    
    param_grid_rf = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1,2,4],
                'bootstrap': [True, False]}

    param_grid_en = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}

    if meth == 'KR':
        gs = GridSearch(model = KernelRidge(), param_grid = param_grid_kr, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        rgr = KernelRidge(kernel=gs.best_params['kernel'], alpha=gs.best_params['alpha'], gamma=gs.best_params['gamma'])
    elif meth == 'RF':        
        gs = GridSearch(model = RandomForestRegressor(random_state=2), param_grid = param_grid_rf, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        rgr = RandomForestRegressor(n_estimators=gs.best_params['n_estimators'], max_features = gs.best_params['max_features'],
                                        min_samples_split=gs.best_params['min_samples_split'],
                                        min_samples_leaf=gs.best_params['min_samples_leaf'], bootstrap=gs.best_params['bootstrap'], 
                                        random_state=2)
    elif meth == 'EN':        
        gs = GridSearch(model = ElasticNet(random_state=2), param_grid = param_grid_en, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        rgr = ElasticNet(alpha = gs.best_params['alpha'], l1_ratio=gs.best_params['l1_ratio'], random_state=2)
    else:
        print('Check Valid Classifier Names')                  

    if trd == 'tr': # this is correct
        rgr.fit(x_tr, y_tr)
    elif trd == 'tr_val':
        rgr.fit(np.concatenate((x_tr, x_va)), np.concatenate((y_tr, y_va)))
    else:
        print('Choose trd as tr or tr_val')
    
    y_tr_pr = rgr.predict(x_tr)
    y_te_pr = rgr.predict(x_te)
    
    return y_tr_pr, y_te_pr

def generate_validation_model_regression(iter_,tr_smp_sizes,nReps,ssd,bs,nw,cuda_avl,mt,lr,ml,es,es_pat,es_va,nc,scorename):                              
    #mode = 'tr'
    #weights = gen_weights(mode,iter_,tr_smp_sizes,nReps,ssd)
    
    if mt == 'AlexNet3D':
        net = AlexNet3D(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=nc)
    elif mt == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=nc)
    else:
        print('Check model type')

    criterion = nn.CrossEntropyLoss()
    
    if mt == 'AlexNet3D_Dropout_Regression':
        #criterion = nn.MSELoss(reduce=False)
        criterion = nn.MSELoss()
    
    trainset = ut.MRIDataset_R(iter_, tr_smp_sizes, nReps, 'tr', ssd, scorename)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)

    validset = ut.MRIDataset_R(iter_, tr_smp_sizes, nReps, 'va', ssd, scorename)
    validloader = DataLoader(validset, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)

    if cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True)

    # Early stopping details
    min_val_mae = 100
    epochs_no_improve = 0
    valid_mae = 0

    history = pd.DataFrame(columns=['iter','epoch','tr_mae','tr_ev','tr_mse','tr_r2','val_mae','val_ev','val_mse','val_r2','loss'])
    
    for epoch in range(es):

        # Train
        print('Training:')
        #loss = train(trainloader, net, optimizer, criterion, cuda_avl, weights)
        loss = train(trainloader, net, optimizer, criterion, cuda_avl)
        loss = loss.data.cpu().numpy()
        y_true, y_pred = test_reg(trainloader, net, cuda_avl)
        train_mae = mean_absolute_error(y_true, y_pred)
        train_ev = explained_variance_score(y_true, y_pred)
        train_mse = mean_squared_error(y_true, y_pred)
        train_r2 = r2_score(y_true, y_pred)

        # Validate
        print('Validating')
        prev_mae = valid_mae
        y_true, y_pred = test_reg(validloader, net, cuda_avl)
        valid_mae = mean_absolute_error(y_true, y_pred)
        valid_ev = explained_variance_score(y_true, y_pred)
        valid_mse = mean_squared_error(y_true, y_pred)
        valid_r2 = r2_score(y_true, y_pred)

        history.loc[epoch] = [iter_, epoch, train_mae, train_ev, train_mse, train_r2, valid_mae, valid_ev, valid_mse, valid_r2, loss]
        history.to_csv(ml + 'history.csv', index=False)

        print ('Iter '+str(iter_)+' Epoch '+str(epoch)+' Tr. MAE.: '+ str(train_mae)+' Tr. EV: '+ str(train_ev)+' Val. MAE: '+str(valid_mae)+' Val. EV : '+str(valid_ev)+' Tr. Loss '+str(loss))

        if es_va:
            
            # If the validation error is at a minimum
            if valid_mae < min_val_mae:
                # Save the model
                torch.save(net.state_dict(), open(ml + 'model_state_dict.pt', 'wb'))
                epochs_no_improve = 0
                min_val_mae = valid_mae
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == es_pat:
                    print('Early stopping!')                    
                    return history, min_val_mae          
        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_mae)

def forward_pass_reg(X_te,net):
    # used to overcome memory constraints
    net.eval()
    outs30=[]
    ims = X_te.shape
    for n in range(0,ims[0]):    
        im = X_te[n].reshape(1,1,ims[2],ims[3],ims[4])          
        temp = net(im)
        temp0 = temp[0].data.cpu().numpy().squeeze()
        outs30.append(temp0)        
    probs = np.vstack((outs30)).squeeze()
    return probs

def evaluate_test_accuracy_regressor(iter_,tr_smp_sizes,nReps,mode,ssd,ml,mt,nc,t0,scorename):

    # Load validated model
    net=0    

    if mt == 'AlexNet3D':
        net = AlexNet3D(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=nc)
    elif mt == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=nc)
    else:
        print('Check model type')

    model = torch.nn.DataParallel(net)
    net = 0
    net = load_net_weights2(model, ml+'model_state_dict.pt')
    
    df_te = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
    X_te, y_te = ut.read_X_y_5D(df_te,scorename)     
    ims = X_te.shape
    X_te = Variable(torch.from_numpy(X_te))

    # save accuracy as csv file
    outs = pd.DataFrame(columns=['iter_','mae_te','ev_te','mse_te','r2_te','Time_Clx','r_te','p_te'])    
    p_y_te = forward_pass_reg(X_te,net)    
  
    """
    # Fit with polyfit
    r_te, p_te = pearsonr(y_te, p_y_te)
    fname = ml+'testscatter.pdf'
    b, m = polyfit(y_te, p_y_te, 1)
    plt.plot(y_te, p_y_te, '.')
    plt.plot(y_te, b + m * y_te, '-')
    plt.title('r='+str(r_te)+'; p='+str(p_te))
    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    """
    mae_te = mean_absolute_error(y_te,p_y_te)
    ev_te = explained_variance_score(y_te,p_y_te)
    mse_te = mean_squared_error(y_te,p_y_te)
    r2_te = r2_score(y_te,p_y_te)

    # write test accuracy and clx time
    t1 = time.time()
    time_clx = t1 - t0
    outs.loc[0] = [iter_, mae_te, ev_te, mse_te, r2_te, time_clx,r_te, p_te ]
    outs.to_csv(ml+'test.csv', index=False)

    return outs, iter_, mae_te

def forward_pass_embeddings(X_te,net,mode):
    # used to overcome memory constraints
    net.eval()
    outs30=[]
    ims = X_te.shape
    for n in range(0,ims[0]):  
        im = X_te[n].reshape(1,1,ims[2],ims[3],ims[4])          
        temp = net.module.features(im)
        temp = temp.view(temp.size(0), -1)
        if mode == 'clx':
            temp = net.module.classifier[1](net.module.classifier[0](temp))            
        elif mode == 'reg':
            temp = net.module.regressor[1](net.module.regressor[0](temp))
        else:
            print('Review mode (reg/clx)')
        outs30.append(temp.data.cpu().numpy().squeeze())  
    probs = np.vstack((outs30))
    return probs

def returnModel(iter_,tr_smp_sizes,nReps,ml,mt,nc):
    
    # Load validated model
    net=0    

    if mt == 'AlexNet3D':
        net = AlexNet3D(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=nc)     
    elif mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_InstanceNorm_SmallConv':
        net = AlexNet3D_Dropout_InstanceNorm_SmallConv(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_InstanceNorm':
        net = AlexNet3D_Dropout_InstanceNorm(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_ProjsDropLin':
        net = AlexNet3D_Dropout_ProjsDropLin(num_classes=nc)
    elif mt == 'AlexNet3D_Dropout_ProjsLin':
        net = AlexNet3D_Dropout_ProjsLin(num_classes=nc)   
    else:
        print('Check model type')

    model = torch.nn.DataParallel(net)
    net = 0
    
    """
    # Use if need to match iter with sample size and rep num
    tv, rv = np.meshgrid(tr_smp_sizes, np.arange(nReps))
    tv = tv.reshape((1, np.prod(tv.shape)))
    rv = rv.reshape((1, np.prod(tv.shape)))
    tss = tv[0][iter_]
    rep = rv[0][iter_]

    net = load_net_weights2(model, ml+str(tss)+'_'+str(rep)+'/model_state_dict.pt')
    """
    net = load_net_weights2(model, ml+'/model_state_dict.pt')    
    
    return net

def sensitivity_analysis(model, image_tensor, target_class=None, postprocess='abs', apply_softmax=True, cuda=False, verbose=False, taskmode='clx'):
    # Adapted from http://arxiv.org/abs/1808.02874
    # https://github.com/jrieke/cnn-interpretability
    
    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square'")
    
    # Forward pass.
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor        print(image_tensor.shape)
        
    if cuda:
        image_tensor = image_tensor.cuda()
    X = Variable(image_tensor[None], requires_grad=True)  # add dimension to simulate batch
    
    output = model(X)[0]    
    
    if apply_softmax:
        output = F.softmax(output)
    
    #print(output.shape)
    
    # Backward pass.
    model.zero_grad()
    
    if taskmode == 'reg':
        output.backward(gradient=output)
    elif taskmode == 'clx':     
        output_class = output.max(1)[1].data[0]
        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
        one_hot_output = torch.zeros(output.size())
        if target_class is None:
            one_hot_output[0, output_class] = 1
        else:
            one_hot_output[0, target_class] = 1
        if cuda:
            one_hot_output = one_hot_output.cuda()        
        output.backward(gradient=one_hot_output)
        
    relevance_map = X.grad.data[0].cpu().numpy()
    
    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map

def area_occlusion(model, image_tensor, area_masks, target_class=None, occlusion_value=0, apply_softmax=True, cuda=False, verbose=False, taskmode='clx'):
    
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor    
    
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))[0]
    
    if apply_softmax:
        output = F.softmax(output)
    
    if taskmode == 'reg':
        unoccluded_prob = output.data
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data.cpu().numpy()[0]    

        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
        
        if target_class is None:
            target_class = output_class
        unoccluded_prob = output.data[0, target_class]
    
    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()
    
    for area_mask in area_masks:

        area_mask = torch.FloatTensor(area_mask)

        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)
        
        output = model(Variable(image_tensor_occluded[None], requires_grad=False))[0]
        if apply_softmax:
            output = F.softmax(output)
            
        if taskmode == 'reg':
            occluded_prob = output.data
        elif taskmode == 'clx':
            occluded_prob = output.data[0, target_class]
        
        ins = area_mask.view(image_tensor.shape) == 1
        ins = ins.squeeze()
        relevance_map[ins] = (unoccluded_prob - occluded_prob)

    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map

def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())
    
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr

def save_nifti(file_path, struct_arr):
    """Save a 3D array to a NIFTI file."""
    img = nib.Nifti1Image(struct_arr, np.eye(4))
    nib.save(img, file_path)

def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)

def get_brain_area_masks(data_size):
    
    brain_map = load_nifti('./aal.nii.gz')
    brain_areas = np.unique(brain_map)[1:]  # omit background

    area_masks = []
    for area in brain_areas:
        area_mask = np.zeros_like(brain_map)
        area_mask[brain_map == area] = 1
        area_mask = resize_image(area_mask, data_size, interpolation=0)
        area_masks.append(area_mask)

    area_names = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
    merged_area_names = [name[:-2] for name in area_names[:108:2]] + area_names[108:]
    
    return area_masks, area_names, merged_area_names

def get_relevance_per_area(area_masks, relevance_map, normalize=True):
    relevances = np.zeros(len(area_masks))
    for i, area_mask in enumerate(area_masks):
        relevances[i] = np.sum(relevance_map * area_mask)
    if normalize:
        relevances /= relevances.sum()  # make all areas sum to 1

    # Merge left and right areas.
    merged_relevances = np.concatenate([relevances[:108].reshape(-1, 2).sum(1), relevances[108:]])

    #return sorted(zip(merged_area_names, merged_relevances), key=lambda b:b[1], reverse=True)
    #return sorted(zip(area_names, relevances), key=lambda b:b[1], reverse=True)
    return relevances

def get_relevance_per_area_norm(area_masks, relevance_map, normalize=True):
    relevances = np.zeros(len(area_masks))
    for i, area_mask in enumerate(area_masks):
        relevances[i] = np.sum(relevance_map * area_mask) / np.sum(area_mask)
    if normalize:
        relevances /= relevances.sum()  # make all areas sum to 1

    # Merge left and right areas.
    merged_relevances = np.concatenate([relevances[:108].reshape(-1, 2).sum(1), relevances[108:]])

    #return sorted(zip(merged_area_names, merged_relevances), key=lambda b:b[1], reverse=True)
    #return sorted(zip(area_names, relevances), key=lambda b:b[1], reverse=True)
    return relevances

def run_saliency(odir, itrpm, images, net, area_masks, iter_, scorename, taskM):
    for nSub in np.arange(images.shape[0]): 
        print(nSub)
        fname = odir + itrpm + '_' + scorename + '_iter_' + str(iter_) + '_nSub_' + str(nSub) + '.nii'    
        if itrpm == 'AO':
            interpretation_method = area_occlusion
            sal_im = interpretation_method(net, images[nSub], area_masks, occlusion_value=0, apply_softmax=False, cuda=False, verbose=False,taskmode=taskM) 
        elif itrpm == 'BP':
            interpretation_method = sensitivity_analysis
            sal_im = interpretation_method(net, images[nSub], apply_softmax=False, cuda=False, verbose=False, taskmode=taskM)
        else:
            print('Verify interpretation method')    
        nib.save(nib.Nifti1Image(sal_im.squeeze() , np.eye(4)), fname)

