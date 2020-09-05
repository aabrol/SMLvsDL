# sbatch --array=[0-59]%60 JSA_SML_reg.sh
import utils as ut
import numpy as np
import pandas as pd
import os
import argparse
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='SML_reg')
parser.add_argument('--ml', default='./results_SML_reg/', metavar='S', help='specify model location')
parser.add_argument('--dataset', default='UKBB', metavar='S', help='specify dataset UKBB or ADNI')
parser.add_argument('--scorename', default='age/', metavar='S', help='specify variable to predict')
args = parser.parse_args()

scorename = args.scorename
out_dir = args.ml
try:
    os.stat(out_dir)
except:
    os.mkdir(out_dir)  
    
if dataset == 'UKBB':
    tr_smp_sizes = [10000]
    task = 'AgeReg'
elif dataset == 'ADNI:
    task = 'REG_MMSE'
    tr_smp_sizes = [428]
else:
    print('Check Dataset')

nReps = 20

dr_meths = ['UFS','RFE','GRP']
reg_meths = ['EN','KR','RF']

tv, rv, dv = np.meshgrid(tr_smp_sizes, np.arange(nReps), dr_meths)
tv = tv.reshape((1, np.prod(tv.shape)))
rv = rv.reshape((1, np.prod(tv.shape)))
dv = dv.reshape((1, np.prod(tv.shape)))

iter_= int( os.environ['SLURM_ARRAY_TASK_ID'] )
tss = tv[0][iter_]
rep = rv[0][iter_]
drm = dv[0][iter_]

t0 = time.time()
x_tr,x_va,x_te,y_tr,y_va,y_te = ut.readFeaturesLabels(task,tss,rep,drm)
t1 = time.time()
time_data_read = t1 - t0
print('Features and Lables Read ..', iter_, task, tss, rep, drm)

outs = pd.DataFrame(columns=['pp','scorename', 'iter_','task','tss','rep','drm', 'reg_meths', 'tr_mae','tr_mse','te_mae','te_mse','time_data_read','time_rgr', 'r_te', 'p_te'])

for pp in np.arange(2):    
    for i in np.arange(len(reg_meths)):         

        # run regressor
        t0 = time.time()
        y_tr_pr, y_te_pr = ut.run_SML_Regressors(reg_meths[i], x_tr, y_tr, x_va, y_va, x_te, y_te, 'tr', pp)  
        t1 = time.time()
        time_rgr = t1 - t0

        # evaluate metrics
        tr_mae = mean_absolute_error(y_tr, y_tr_pr)
        tr_mse = mean_squared_error(y_tr, y_tr_pr)
        te_mae = mean_absolute_error(y_te, y_te_pr)
        te_mse = mean_squared_error(y_te, y_te_pr)
        
        """
        # Fit with polyfit
        r_te, p_te = pearsonr(y_te, y_te_pr)
        fname = out_dir + scorename + '_' + reg_meths[i] + '_' + drm + '_pp_' + str(pp) + 'rep' + str(rep) + 'testscatter.pdf'
        b, m = polyfit(y_te, y_te_pr, 1)
        plt.plot(y_te, y_te_pr, '.')
        plt.plot(y_te, b + m * y_te, '-')
        plt.title('r='+str(r_te)+'; p='+str(p_te))
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        
        plt.clf()
        """

        # write results
        outs.loc[i] = [pp,scorename,iter_,task,tss,rep,drm,reg_meths[i],tr_mae,tr_mse,te_mae,te_mse,time_data_read,time_rgr, r_te, p_te]
        outs.to_csv(out_dir + scorename + '_' + reg_meths[i] + '_' + drm + '_pp_' + str(pp) + 'rep' + str(rep) + '.csv', index=False)
        
