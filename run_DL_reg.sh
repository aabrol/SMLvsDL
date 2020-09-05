#!/bin/bash

# Age Regression
python run_DL_reg.py --scorename 'age' --nc 1 --bs 32 --lr 0.001 --es 500 --es_va 1 --es_pat 40 --ml './results_DL_reg_Age/' --ssd './SampleSplits_Age/' --mt 'AlexNet3D_Dropout_Regression' --dataset 'UKBB' &

# MMSE Regression
# python run_DL_reg.py --scorename 'MMSE' --nc 1 --bs 32 --lr 0.001 --es 500 --es_va 1 --es_pat 40 --ml './results_DL_reg_MMSE/' --ssd './SampleSplits_MMSE/'  --mt 'AlexNet3D_Dropout_Regression' --dataset 'ADNI' &

wait
