# sbatch --array=[0-1259]%60 JSA_SML.sh
import utils as ut
import numpy as np
import pandas as pd
import os
import argparse
import time

parser = argparse.ArgumentParser(description='SML')
parser.add_argument('--mi', type=int, default = 0, metavar='N', help='max. iterations for classifiers: default = 1000')
parser.add_argument('--ml', default='./results/', metavar='S', help='model location (default: ./results/)')
args = parser.parse_args()

mi = args.mi
out_dir = args.ml
try:
    os.stat(out_dir)
except:
    os.mkdir(out_dir)  
    
tasks = ['AgeSex','Age','Sex']
tr_smp_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
nReps = 20
dr_meths = ['UFS','RFE','GRP']
clx_meths = ['LDA','LR','SVML','SVMR','SVMP','SVMS']

# Generate a meshgrid to parallelize jobs for slurm
# Mesh over (a) Sample Size (n=7) (b) Repetetions (n=20) (c) Tasks (d) Dimension Reduction Methods
# Hence 7*3*20*3=1260 Jobs
tv, taskv, rv, dv = np.meshgrid(tr_smp_sizes, tasks, np.arange(nReps), dr_meths)
taskv = taskv.reshape((1, np.prod(tv.shape)))
tv = tv.reshape((1, np.prod(tv.shape)))
rv = rv.reshape((1, np.prod(tv.shape)))
dv = dv.reshape((1, np.prod(tv.shape)))

# Get slurm env. variables (0-1259) and run job
iter_= int( os.environ['SLURM_ARRAY_TASK_ID'] )
task = taskv[0][iter_]
tss = tv[0][iter_]
rep = rv[0][iter_]
drm = dv[0][iter_]

t0 = time.time()
x_tr,x_va,x_te,y_tr,y_va,y_te = ut.readFeaturesLabels(task,tss,rep,drm,scorename)
t1 = time.time()
time_data_read = t1 - t0
print('Features and Lables Read ..', iter_, task, tss, rep, drm)

outs = pd.DataFrame(columns=['postproc','Iter','Task','SampSize','Rep','DRM', 'Clx', 'TeAcc','Time_Data_Read','Time_Clx'])
for pp in np.arange(2):
    for i in np.arange(len(clx_meths)): 
        t0 = time.time()
        scr = ut.run_SML_Classifiers(clx_meths[i], x_tr, y_tr, x_va, y_va, x_te, y_te, 'tr', pp, mi)
        t1 = time.time()
        time_classifier = t1 - t0
        outs.loc[i] = [pp,iter_,task,tss,rep,drm,clx_meths[i],scr,time_data_read,time_classifier]
        outs.to_csv(out_dir+'outs_pp_'+str(pp)+'_mi_'+str(mi)+'_iter_'+str(iter_)+'.csv', index=False)
