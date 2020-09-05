# sbatch --array=[0-139] JSA_DR.sh

import os
import utils as ut
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_csv_fn = './AA_UKB.csv'
df = pd.read_csv(df_csv_fn) # Contains 12314 Subject fields [eid,fmriPath,smriPath,age,sex,label,age5label]
# Here age, sex and label are labels for the age regression, sex classification and age-and-sex-based classification tasks respectively.
# Train Size: [100 200 500 1000 2000 5000 10000] Val Size: 2314/2 = 1157 Test Size: 2314/2 = 1157
mskFn = '/data/mialab/competition2019/UKBiobank/process/results/smri_mask_generation/ica_analysisMask.nii'

tr_smp_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
nReps = 20
nJobs = len(tr_smp_sizes)*nReps
nFeats = 784
#tasks = ['AgeSex','Age','Sex']
scores = ['label','sex','age5label']
taskDirs = ['./SampleSplits', './SampleSplits_Sex', './SampleSplits_Age']
featDirs = ['./sMRI_feats', './sMRI_feats_Sex' , './sMRI_feats_Age']

# Generate a meshgrid to parallelize jobs for slurm
# Mesh over (a) Sample Size (n=7) (b) Repetetions (n=20)
# Hence 20*7=140 Jobs
xv, yv = np.meshgrid(tr_smp_sizes, np.arange(nReps))
xv = xv.reshape((1, nJobs)) 
yv = yv.reshape((1, nJobs)) 

iter_= int( os.environ['SLURM_ARRAY_TASK_ID'] )
tss = xv[0][iter_]
rep = yv[0][iter_]

for score, tdir, fdir in zip(scores, taskDirs, featDirs):
    
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir) 
  
    print( tss, rep )

    # A. Read Subject Splits for Each Training Sample Size and Repetition
    df_tr = pd.read_csv(tdir + '/tr_' + str(tss) + '_rep_' + str(rep) + '.csv')
    df_va = pd.read_csv(tdir + '/va_' + str(tss) + '_rep_' + str(rep) + '.csv')
    df_te = pd.read_csv(tdir + '/te_' + str(tss) + '_rep_' + str(rep) + '.csv')
    print('Tr:'+str(df_tr.shape)+' Va:'+str(df_va.shape)+' Te:'+str(df_te.shape))        
    print('DataFrames Read ...')

    # B. Read Scans
    X_tr, y_tr = ut.read_X_y(df_tr, mskFn, score)
    X_va, y_va = ut.read_X_y(df_va, mskFn, score)
    X_te, y_te = ut.read_X_y(df_te, mskFn, score)        
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
    np.savetxt(fdir + '/X_tr_UFS' + str(tss) + '_rep_' + str(rep) + '.csv', X_tr_UFS, delimiter=',', fmt='%f')
    np.savetxt(fdir + '/X_va_UFS' + str(tss) + '_rep_' + str(rep) + '.csv', X_va_UFS, delimiter=',', fmt='%f')
    np.savetxt(fdir + '/X_te_UFS' + str(tss) + '_rep_' + str(rep) + '.csv', X_te_UFS, delimiter=',', fmt='%f')
    print('UFS Dim. Red. Done ... ')
    X_tr_UFS, X_va_UFS, X_te_UFS = [],[],[]

    X_tr_RFE, X_va_RFE, X_te_RFE = ut.red_dim(X_tr, y_tr, X_va, X_te, nFeats, 'RFE')
    np.savetxt(fdir + '/X_tr_RFE' + str(tss) + '_rep_' + str(rep) + '.csv', X_tr_RFE, delimiter=',', fmt='%f')
    np.savetxt(fdir + '/X_va_RFE' + str(tss) + '_rep_' + str(rep) + '.csv', X_va_RFE, delimiter=',', fmt='%f')
    np.savetxt(fdir + '/X_te_RFE' + str(tss) + '_rep_' + str(rep) + '.csv', X_te_RFE, delimiter=',', fmt='%f')
    print('RFE Dim. Red. Done ... ')
    X_tr_RFE, X_va_RFE, X_te_RFE = [],[],[]

    X_tr_GRP, X_va_GRP, X_te_GRP = ut.red_dim(X_tr, y_tr, X_va, X_te, nFeats, 'GRP')
    np.savetxt(fdir + '/X_tr_GRP' + str(tss) + '_rep_' + str(rep) + '.csv', X_tr_GRP, delimiter=',', fmt='%f')
    np.savetxt(fdir + '/X_va_GRP' + str(tss) + '_rep_' + str(rep) + '.csv', X_va_GRP, delimiter=',', fmt='%f')
    np.savetxt(fdir + '/X_te_GRP' + str(tss) + '_rep_' + str(rep) + '.csv', X_te_GRP, delimiter=',', fmt='%f')
    print('GRP Dim. Red. Done ... ')
    X_tr_GRP, X_va_GRP, X_te_GRP = [],[],[]

    print('Sample Size: ', tss, ' Rep: ', rep, ' Features Saved ... ')
