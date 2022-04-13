# sbatch --array=[0-139:2]%8 JSA_DL.sh

import os
import utils as ut
import torch
import argparse
import time
import numpy as np


parser = argparse.ArgumentParser(description='SLvsML')
parser.add_argument('--nc', type=int, default=10,metavar='N',
                    help='number of classes in dataset (default: 10)')
parser.add_argument('--bs', type=int, default=32, metavar='N',
          help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', 
  help='learning rate (default: 0.001)')
parser.add_argument('--es', type=int, default=100, metavar='N',
          help='number of epochs to train (default: 100)')
parser.add_argument('--pp', type=int, default = 0, metavar='N',
          help='iteration flag to allow multiple  processes')
parser.add_argument('--es_va', type=int, default = 1, metavar='N',
          help='1: use val accuracy; 0: use val loss : default on i.e. uses val accuracy')
parser.add_argument('--es_pat', type=int, default=20, metavar='N',
          help='patience for early stopping (default: 20)')
parser.add_argument('--ml', default='./temp/', metavar='S',
          help='model location (default: ./temp/)')
parser.add_argument('--mt', default='AlexNet3D', metavar='S',
                   help = 'modeltype (default: AlexNet3D)')
parser.add_argument('--ssd', default='/SampleSplits/', metavar='S',
          help='sample size directory (default: /SampleSplits/)')
parser.add_argument('--scorename', default='age', metavar='S',
          help='scorename (default: fluid_intelligence_score_f20016_2_0)')
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
bs = args.bs
lr = args.lr
es = args.es
pp = args.pp
es_va = args.es_va
es_pat = args.es_pat
ssd = args.ssd
nw = 2
mt = args.mt
cuda_avl = args.cuda
scorename = args.scorename

iter_ = int(os.environ['SLURM_ARRAY_TASK_ID'])
if pp:
    iter_ += 1
ml = args.ml+str(iter_)+'/'
try:
    os.stat(ml)
except:
    os.mkdir(ml)  

tr_smp_sizes = [100, 200, 500, 1000, 2000, 5000 , 10000]
nReps = 20

t0 = time.time()
# Train/Validate
history, max_val_acc = ut.generate_validation_model(iter_,tr_smp_sizes,nReps,ssd,bs,nw,cuda_avl,mt,lr,ml,es,es_pat,es_va,nc,scorename)
# Test
mode = 'te'
outs, iter_, acc_te = ut.evaluate_test_accuracy(iter_,tr_smp_sizes,nReps,mode,ssd,ml,mt,nc,t0,scorename)
