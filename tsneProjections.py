import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import utils as ut
from models import AlexNet3D_Dropout, AlexNet3D_Dropout_Regression
import torch
from torch.autograd import Variable

######### AGE-GENDER-BASED CLASSIFICATION ########

# Parameter list
scorename = 'label'
nc = 10
ssd = './SampleSplits/'
tr_smp_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
nReps = 20
mode = 'te'
cmp = ['#a50026','#d73027','#f46d43','#fdae61','#fee08b','#053061','#2166ac','#4393c3','#92c5de','#d1e5f0']
lbs = ['F:45-52','M:45-52','F:53-59','M:53-59','F:60-66','M:60-66','F:67-73','M:67-73','F:74-80','M:74-80']
lo = [0,2,4,6,8,1,3,5,7,9] # Reorder to match gender groups with reds and blues

rep = 0 #First repetition of all 7 sample sizes
for iter_ in arange(7): 

    # Specify model location
    ml = './results_DL_AgeSexC/'
    ml = ml+str(iter_)+'/'

    # Read model
    net = AlexNet3D_Dropout(num_classes=nc)
    model = torch.nn.DataParallel(net)
    net = ut.load_net_weights2(model, ml+'model_state_dict.pt')

    # Read labels
    labels = pd.read_csv(ssd+'te_'+str(tr_smp_sizes[iter_])+'_rep_'+str(rep) + '.csv')[scorename].values
    u_labels = np.unique(labels)
    u_labels = np.unique(labels)[lo]

    # Read data
    df_te = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
    X_te, y_te = ut.read_X_y_5D(df_te,scorename)     
    X_te = Variable(torch.from_numpy(X_te))

    # Forward Pass (Generate Embeddings)
    embs = ut.forward_pass_embeddings(X_te,net,'clx')

    # Project Embeddings
    X_embs = TSNE(n_components=2, perplexity=30, learning_rate=100, random_state=1).fit_transform(embs)

    # Plot Spectra
    plt.figure(num=1, figsize=(6, 4), dpi=500, facecolor='w', edgecolor='w')
    i=0
    for ul in u_labels:
        plt.scatter(X_embs[labels==ul,0], X_embs[labels==ul, 1],label=lbs[ul-1],s=7,c=cmp[i],cmap=cmp)
        i+=1
    plt.axis('tight')
    plt.yticks([])
    plt.xticks([])
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2,fancybox=True, shadow=False, fontsize = 6)
    #plt.show()
    plt.savefig('./Figures/age_gen_combined_clx_projection_'+str(tr_smp_sizes[iter_]), dpi = 300)
    plt.clf()

######### GENDER CLASSIFICATION ########

# Parameter list
scorename = 'sex' # Sex Regression
iter_ = 6 # First repetition of largest of 7 sample sizes
nc = 2
ssd = './SampleSplits_Sex/'
tr_smp_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
nReps = 20
mode = 'te'
cmp = ['#f46d43','#4393c3']
lbs = ['F','M']

# Specify model location
ml = './results_DL_SexC/'
ml = ml+str(iter_)+'/'

# Read model
net = AlexNet3D_Dropout(num_classes=nc)
model = torch.nn.DataParallel(net)
net = ut.load_net_weights2(model, ml+'model_state_dict.pt')

# Read labels
labels = pd.read_csv(ssd+'te_'+str(tr_smp_sizes[6])+'_rep_'+str(iter_) + '.csv')[scorename].values
u_labels = np.unique(labels)

# Read data
df_te = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
X_te, y_te = ut.read_X_y_5D(df_te,scorename)     
X_te = Variable(torch.from_numpy(X_te))

# Forward Pass (Generate Embeddings)
embs = ut.forward_pass_embeddings(X_te,net,'clx')

# Project Embeddings
X_embs = TSNE(n_components=2, perplexity=40, learning_rate=500, random_state=1).fit_transform(embs)

# Plot Spectra
plt.figure(num=1, figsize=(6, 4), dpi=500, facecolor='w', edgecolor='w')
for ul in u_labels:
    plt.scatter(X_embs[labels==ul,0], X_embs[labels==ul, 1], label=lbs[ul-1],s=7,c=cmp[ul-1],cmap=cmp)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.legend(loc=9, bbox_to_anchor=(0.5, 0.4), ncol=2,fancybox=True, shadow=False, fontsize = 10)
plt.show()
plt.savefig('./Figures/gen_clx_projection', dpi = 300)
plt.clf()

###### AGE REGRESSION ##########

# Parameter list
scorename = 'age' # Age Regression
mt = 'AlexNet3D_Dropout_Regression'
es_pat = 40
iter_ = 0 # First crossvalidation repetition
lr = 0.001
nc = 1
cmp = 'winter'
ssd = './SampleSplits_Age/'
tr_smp_sizes = [10000]
nReps = 20
mode = 'te'

# Specify model location
ml = './results_DL_reg_Age/'
ml = ml+scorename+'_'+str(mt)+'_pat_'+str(es_pat)+'_iter_'+str(iter_)+'_lr_'+str(lr)+'/'

# Read model
net = AlexNet3D_Dropout_Regression(num_classes=nc)
model = torch.nn.DataParallel(net)
net = ut.load_net_weights2(model, ml+'model_state_dict.pt')

# Read labels
labels = pd.read_csv(ssd+'te_'+str(tr_smp_sizes[0])+'_rep_'+str(iter_) + '.csv')[scorename].values

# Read data
df_te = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
X_te, y_te = ut.read_X_y_5D(df_te,scorename)     
X_te = Variable(torch.from_numpy(X_te))

# Forward Pass (Generate Embeddings)
embs = ut.forward_pass_embeddings(X_te,net,'reg')

# Project Embeddings
X_embs = TSNE(n_components=2, perplexity=100, learning_rate=300, random_state=1).fit_transform(embs)

# Plot Spectra
plt.figure(num=1, figsize=(6, 4), dpi=500, facecolor='w', edgecolor='w')
plt.scatter(X_embs[:,0], X_embs[:, 1], c=labels, s=2, cmap=cmp)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
#plt.legend(loc=9, bbox_to_anchor=(0.5, 0.4), ncol=2, shadow=False, fontsize = 10)
cbar = plt.colorbar(orientation='horizontal', fraction = 0.05)
cbar.set_label('Age')
#plt.show()
plt.savefig('./Figures/age_reg_projection', dpi = 300)
plt.clf()

######### MMSE REGRESSION ######

# Parameter list
scorename = 'MMSE' # Age Regression
mt = 'AlexNet3D_Dropout_Regression'
es_pat = 40
iter_ = 0 # First crossvalidation repetition
lr = 0.001
nc = 1
cmp = 'winter'
ssd = './SampleSplits_MMSE/'
tr_smp_sizes = [428]
nReps = 20
mode = 'te'

# Specify model location
ml = './results_DL_reg_MMSE/'
ml = ml+scorename+'_'+str(mt)+'_pat_'+str(es_pat)+'_iter_'+str(iter_)+'_lr_'+str(lr)+'/'

# Read model
net = AlexNet3D_Dropout_Regression(num_classes=nc)
model = torch.nn.DataParallel(net)
net = ut.load_net_weights2(model, ml+'model_state_dict.pt')

# Read labels
labels = pd.read_csv(ssd+'te_'+str(tr_smp_sizes[0])+'_rep_'+str(iter_) + '.csv')[scorename].values

# Read data
df_te = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
X_te, y_te = ut.read_X_y_5D(df_te,scorename)     
X_te = Variable(torch.from_numpy(X_te))

# Forward Pass (Generate Embeddings)
embs = ut.forward_pass_embeddings(X_te,net,'reg')

# Project Embeddings
X_embs = TSNE(n_components=2, perplexity=70, learning_rate=200, random_state=1).fit_transform(embs)

# Plot Spectra
plt.figure(num=1, figsize=(6, 4), dpi=500, facecolor='w', edgecolor='w')
plt.scatter(X_embs[:,0], X_embs[:, 1], c=labels, s=2, cmap=cmp)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
#plt.legend(loc=9, bbox_to_anchor=(0.5, 0.4), ncol=2, shadow=False, fontsize = 10)
cbar = plt.colorbar(orientation='horizontal', fraction = 0.05)
cbar.set_label('MMSE')
#plt.show()
plt.savefig('./Figures/mmse_reg_projection', dpi = 300)
plt.clf()
