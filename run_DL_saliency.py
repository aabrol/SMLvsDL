import os
import numpy as np
import utils as ut
import torch
import argparse
import time

parser = argparse.ArgumentParser(description='SLvsML_Saliency')
parser.add_argument('--nc', type=int, default=10,metavar='N',
                    help='number of classes in dataset (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', 
  help='learning rate (default: 0.0001)')
parser.add_argument('--es_pat', type=int, default=20, metavar='N',
          help='patience for early stopping (default: 20)')
parser.add_argument('--ml', default='./temp/', metavar='S',
          help='model location (default: ./temp/)')
parser.add_argument('--mt', default='AlexNet3D', metavar='S',
                   help = 'modeltype (default: AlexNet3D)')
parser.add_argument('--ssd', default='/SampleSplits/', metavar='S',
          help='sample size directory (default: /SampleSplits/)')
parser.add_argument('--scorename', default='age', metavar='S',
          help='scorename (default: label)')
parser.add_argument('--odir', default='./temp/', metavar='S',
          help='output results location (default: ./temp/)')
parser.add_argument('--itrpm', default='AO', metavar='S',
          help='interpretation mode (default: Area Occlusion)')
parser.add_argument('--taskM', default='clx', metavar='S',
          help='task mode: regression (reg) or classification (clx) (default: clx')
parser.add_argument('--mode', default='te', metavar='S',
          help='data mode: train (tr), validation (va) or test (te) (default: te')
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
lr = args.lr
es_pat = args.es_pat
mt = args.mt
ssd = args.ssd
scorename = args.scorename
odir = args.odir
itrpm = args.itrpm
taskM = args.taskM
mode = args.mode
cuda_avl = args.cuda

iter_ = int( os.environ['SLURM_ARRAY_TASK_ID'] )

ml = os.path.abspath(args.ml)
if taskM == 'reg':
    ml = args.ml + scorename+'_'+str(mt)+'_pat_'+str(es_pat)+'_iter_'+str(iter_)+'_lr_'+str(lr)+'/'
    tr_smp_sizes = [10000]
elif taskM == 'clx':
    ml = args.ml + str(iter_)+'/'
    tr_smp_sizes = [100,200,500,1000,2000,5000,10000]
else:
    print('Review Task Mode')

try:
    os.stat(os.path.abspath(odir))
except:
    os.mkdir(os.path.abspath(odir))  

### Read test data ### 
nReps = 20
df = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
X, y = ut.read_X_y_5D(df,scorename)
dSize = X[0].squeeze().shape

### Load model ###
net = ut.returnModel(iter_,tr_smp_sizes,nReps,ml,mt,nc)
net.eval()

### Save Saliency ###
area_masks, _ , _  = ut.get_brain_area_masks(dSize)
ut.run_saliency(odir, itrpm, X, net, area_masks, iter_, scorename, taskM)
