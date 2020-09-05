#!/bin/bash

# Run classification or regression tasks
# classification (age_gender or gender): sbatch --array=[6-139:7] JSA_DL_saliency.sh (only largest sample sizes) 
# regression (age/mmse):                 sbatch --array=[0-19] JSA_DL_saliency.sh

#AgeSex Classification (10 way)
python run_DL_saliency.py --nc 10 --ml './results_DL_AgeSexC/' --mt 'AlexNet3D_Dropout' --ssd './SampleSplits/' --scorename 'label' --odir './Saliency/' --itrpm 'BP' --taskM 'clx' --mode 'te' 
python run_DL_saliency.py --nc 10 --ml './results_DL_AgeSexC/' --mt 'AlexNet3D_Dropout' --ssd './SampleSplits/' --scorename 'label' --odir './Saliency/' --itrpm 'AO' --taskM 'clx' --mode 'te' 

#Sex Classification (2 way)
python run_DL_saliency.py --nc 2 --ml './results_DL_SexC/' --mt 'AlexNet3D_Dropout' --ssd './SampleSplits_Sex/' --scorename 'sex' --odir './Saliency/' --itrpm 'BP' --taskM 'clx' --mode 'te' 
python run_DL_saliency.py --nc 2 --ml './results_DL_SexC/' --mt 'AlexNet3D_Dropout' --ssd './SampleSplits_Sex/' --scorename 'sex' --odir './Saliency/' --itrpm 'AO' --taskM 'clx' --mode 'te' 

#Age Regression
#python run_DL_saliency.py --nc 1 --lr 0.001 --es_pat 40 --ml './results_DL_reg_Age/' --mt 'AlexNet3D_Dropout_Regression' --ssd './SampleSplits_Age/' --scorename 'age' --odir './Saliency/' --itrpm 'BP' --taskM 'reg' --mode 'te' 
#python run_DL_saliency.py --nc 1 --lr 0.001 --es_pat 40 --ml './results_DL_reg_Age/' --mt 'AlexNet3D_Dropout_Regression' --ssd './SampleSplits_Age/' --scorename 'age' --odir './Saliency/' --itrpm 'AO' --taskM 'reg' --mode 'te' 

#MMSE Regression
#python run_DL_saliency.py --nc 1 --lr 0.001 --es_pat 40 --ml './results_DL_reg_MMSE/' --mt 'AlexNet3D_Dropout_Regression' --ssd './SampleSplits_MMSE/' --scorename 'MMSE' --odir './Saliency/' --itrpm 'BP' --taskM 'reg' --mode 'te' 
#python run_DL_saliency.py --nc 1 --lr 0.001 --es_pat 40 --ml './results_DL_reg_MMSE/' --mt 'AlexNet3D_Dropout_Regression' --ssd './SampleSplits_MMSE/' --scorename 'MMSE' --odir './Saliency/' --itrpm 'AO' --taskM 'reg' --mode 'te' 

wait