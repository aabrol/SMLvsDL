#!/bin/bash

# Run two iterations of one task on same GPU concurrently (pp=0,pp=1)
#AgeSex Classification (10 way)
python run_DL.py --nc 10 --bs 32 --lr 0.00001 --es 200 --pp 0 --es_va 1 --es_pat 20 --ml './results_DL_AgeSexC/' --ssd './SampleSplits/' --mt 'AlexNet3D_Dropout' --scorename 'label' &
python run_DL.py --nc 10 --bs 32 --lr 0.00001 --es 200 --pp 1 --es_va 1 --es_pat 20 --ml './results_DL_AgeSexC/' --ssd './SampleSplits/' --mt 'AlexNet3D_Dropout' --scorename 'label' &

#Sex Classification (2 way)
#python run_DL.py --nc 2 --bs 32 --lr 0.00001 --es 200 --pp 0 --es_va 1 --es_pat 20 --ml './results_DL_SexC/' --ssd './SampleSplits_Sex/' --mt 'AlexNet3D_Dropout' --scorename 'sex' &
#python run_DL.py --nc 2 --bs 32 --lr 0.00001 --es 200 --pp 1 --es_va 1 --es_pat 20 --ml './results_DL_SexC/' --ssd './SampleSplits_Sex/' --mt 'AlexNet3D_Dropout' --scorename 'sex' &

#Age Classification (5 way)
#python run_DL.py --nc 5 --bs 32 --lr 0.00001 --es 200 --pp 0 --es_va 1 --es_pat 20 --ml './results_DL_AgeC/' --ssd './SampleSplits_Age/' --mt 'AlexNet3D_Dropout' --scorename 'age' &
#python run_DL.py --nc 5 --bs 32 --lr 0.00001 --es 200 --pp 1 --es_va 1 --es_pat 20 --ml './results_DL_AgeC/' --ssd './SampleSplits_Age/' --mt 'AlexNet3D_Dropout' --scorename 'age' &

wait
