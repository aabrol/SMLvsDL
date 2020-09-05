# sbatch --array=[0-19] JSA_DR_ADNI.sh

import os
import utils as ut
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./Progression_Master.csv')
mskFn = './mask_gI_828_c.nii'
sdir = './SampleSplits_MMSE/'

odir = './sMRI_feats_MMSE/'
try:
    os.stat(odir)
except:
    os.mkdir(odir) 

tr_smp_sizes = [428]
nReps = 20
nFeats = 784

xv, yv = np.meshgrid(tr_smp_sizes, np.arange(nReps))
xv = xv.reshape((1, nReps)) 
yv = yv.reshape((1, nReps))
iter_= int( os.environ['SLURM_ARRAY_TASK_ID'] )
tss = xv[0][iter_]
rep = yv[0][iter_]
print(tss, rep)

# A. Read Subject Splits for Each Training Sample Size and Repetition
df_tr = pd.read_csv(sdir + 'tr_' + str(tss) + '_rep_' + str(rep) + '.csv')
df_va = pd.read_csv(sdir + 'va_' + str(tss) + '_rep_' + str(rep) + '.csv')
df_te = pd.read_csv(sdir + 'te_' + str(tss) + '_rep_' + str(rep) + '.csv')
print('Tr:'+str(df_tr.shape)+' Va:'+str(df_va.shape)+' Te:'+str(df_te.shape))        
print('DataFrames Read ...')

# B. Read Scans
X_tr, y_tr = ut.read_X_y_ADNI(df_tr, mskFn)
X_va, y_va = ut.read_X_y_ADNI(df_va, mskFn)
X_te, y_te = ut.read_X_y_ADNI(df_te, mskFn)        
print('Images Read ...')

# C. Normalization
ss = StandardScaler().fit(np.concatenate((X_tr, X_va)))
X_tr = ss.transform(X_tr)
X_va = ss.transform(X_va)
X_te = ss.transform(X_te)
print(X_tr.shape, X_va.shape, X_te.shape)
print(np.mean(X_tr), np.mean(X_va), np.mean(X_te))
print('Data Normalized ...')

# D. Dimension Reduction
X_tr_UFS, X_va_UFS, X_te_UFS = ut.red_dim(X_tr, y_tr, X_va, X_te, nFeats, 'UFS')
np.savetxt(odir + 'X_tr_UFS' + str(tss) + '_rep_' + str(rep) + '.csv', X_tr_UFS, delimiter=',', fmt='%f')
np.savetxt(odir + 'X_va_UFS' + str(tss) + '_rep_' + str(rep) + '.csv', X_va_UFS, delimiter=',', fmt='%f')
np.savetxt(odir + 'X_te_UFS' + str(tss) + '_rep_' + str(rep) + '.csv', X_te_UFS, delimiter=',', fmt='%f')
print('UFS Dim. Red. Done ... ')
X_tr_UFS, X_va_UFS, X_te_UFS = [],[],[]

X_tr_RFE, X_va_RFE, X_te_RFE = ut.red_dim(X_tr, y_tr, X_va, X_te, nFeats, 'RFE')
np.savetxt(odir + 'X_tr_RFE' + str(tss) + '_rep_' + str(rep) + '.csv', X_tr_RFE, delimiter=',', fmt='%f')
np.savetxt(odir + 'X_va_RFE' + str(tss) + '_rep_' + str(rep) + '.csv', X_va_RFE, delimiter=',', fmt='%f')
np.savetxt(odir + 'X_te_RFE' + str(tss) + '_rep_' + str(rep) + '.csv', X_te_RFE, delimiter=',', fmt='%f')
print('RFE Dim. Red. Done ... ')
X_tr_RFE, X_va_RFE, X_te_RFE = [],[],[]

X_tr_GRP, X_va_GRP, X_te_GRP = ut.red_dim(X_tr, y_tr, X_va, X_te, nFeats, 'GRP')
np.savetxt(odir + 'X_tr_GRP' + str(tss) + '_rep_' + str(rep) + '.csv', X_tr_GRP, delimiter=',', fmt='%f')
np.savetxt(odir + 'X_va_GRP' + str(tss) + '_rep_' + str(rep) + '.csv', X_va_GRP, delimiter=',', fmt='%f')
np.savetxt(odir + 'X_te_GRP' + str(tss) + '_rep_' + str(rep) + '.csv', X_te_GRP, delimiter=',', fmt='%f')
print('GRP Dim. Red. Done ... ')
X_tr_GRP, X_va_GRP, X_te_GRP = [],[],[]

print('Sample Size: ', tss, ' Rep: ', rep, ' Features Saved ... ')

