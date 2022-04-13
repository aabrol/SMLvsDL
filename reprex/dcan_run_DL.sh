#!/bin/bash -l

#SBATCH --job-name=alex_net # job name
#SBATCH --mem=90g        # memory per cpu-core (what is the default?)
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e alex_net-%j.err
#SBATCH -o alex_net-%j.out

cd /home/miran045/reine097/projects/AlexNet_Abrol2021/reprex || exit
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/reprex/run_DL.py --nc 2 --ssd ./SampleSplits_MMSE \
  --mt AlexNet3D_Dropout --tss 200 --tr_smp_sizes 200 --nw 2 --bs 8

wait