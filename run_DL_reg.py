# sbatch --array=[0-19]%8 JSA_DL_reg.sh
# sbatch --array=[0-19]%8 JSA_DL_reg.sh

import os
import utils as ut
import numpy as np
import torch
import argparse
import time

parser = argparse.ArgumentParser(description='SLvsML')
parser.add_argument('--scorename', default='MMSE', metavar='S',
                   help = 'scorename (default: MMSE)')
parser.add_argument('--nc', type=int, default=10,metavar='N',
                    help='number of classes in dataset (default: 10)')
parser.add_argument('--bs', type=int, default=32, metavar='N',
          help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', 
  help='learning rate (default: 0.001)')
parser.add_argument('--es', type=int, default=100, metavar='N',
          help='number of epochs to train (default: 100)')
parser.add_argument('--es_va', type=int, default = 1, metavar='N',
          help='1: use val accuracy; 0: use val loss : default on i.e. uses val accuracy')
parser.add_argument('--es_pat', type=int, default=20, metavar='N',
          help='patience for early stopping (default: 20)')
parser.add_argument('--ml', default='/temp/', metavar='S',
          help='model location (default: /temp/)')
parser.add_argument('--ssd', default='/SampleSplits/', metavar='S',
          help='sample size directory (default: /SampleSplits/)')
parser.add_argument('--mt', default='AlexNet3D', metavar='S',
                   help = 'modeltype (default: AlexNet3D)')
parser.add_argument('--dataset', default='UKBB', metavar='S', help='specify dataset UKBB or ADNI')
parser.add_argument('--no-cuda', action='store_true', default=False,
          help='turn off to enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
          help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed) 
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# input params
nc = args.nc
scorename = args.scorename
bs = args.bs
lr = args.lr
pp = args.pp
mt = args.mt
nw = 8
ssd = args.ssd
es = args.es
es_va = args.es_va
es_pat = args.es_pat
cuda_avl = args.cuda

iter_ = int(os.environ['SLURM_ARRAY_TASK_ID'])

ml = args.ml+scorename+'_'+str(mt)+'_pat_'+str(es_pat)+'_iter_'+str(iter_)+'_lr_'+str(lr)+'/'
try:
    os.stat(ml)
except:
    os.mkdir(ml)  
    
if dataset == 'UKBB':
    tr_smp_sizes = [10000]
elif dataset == 'ADNI:
    tr_smp_sizes = [428]
else:
    print('Check Dataset')

nReps = 20

t0 = time.time()
# Train/Validate
history, min_val_mae = ut.generate_validation_model_regression(iter_,tr_smp_sizes,nReps,ssd,bs,nw,cuda_avl,mt,lr,ml,es,es_pat,es_va,nc,scorename)
# Test
mode = 'te'
outs, iter_, mae_te = ut.evaluate_test_accuracy_regressor(iter_,tr_smp_sizes,nReps,mode,ssd,ml,mt,nc,t0,scorename)
