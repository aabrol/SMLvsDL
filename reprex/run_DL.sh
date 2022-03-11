#!/bin/bash

# Classification (10 way) - Run two iterations of one task on same GPU concurrently (pp=0,pp=1)
python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits/'  --scorename 'label' --nw 4 --cr 'clx' &

python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 1 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits/'  --scorename 'label' --nw 4 --cr 'clx' &

# Classification (2 way) - Run two iterations of one task on same GPU concurrently (pp=0,pp=1)
#python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 0 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Sex/'  --scorename 'label' --nw 4 --cr 'clx' &

#python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 10 --bs 32 --lr 0.00001 --es 500 --pp 1 --es_va 1 --es_pat 20 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Sex/'  --scorename 'label' --nw 4 --cr 'clx' &

# regression
#python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 0 --es_va 1 --es_pat 40 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Age/'  --scorename 'age' --nw 4 --cr 'reg' &

#python run_DL.py --tr_smp_sizes 100 200 500 1000 2000 5000 10000 --nReps 20 --nc 1 --bs 32 --lr 0.001 --es 500 --pp 1 --es_va 1 --es_pat 40 --ml '../results/' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_Age/'  --scorename 'age' --nw 4 --cr 'reg' &

wait