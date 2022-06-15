#!/bin/sh

#SBATCH --job-name=alexnet.training.agate

#SBATCH --mem=90g
#SBATCH --time=1:00:00
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=6

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e alex-net-training-%j.err
#SBATCH -o alex-net-training-%j.out

#SBATCH -A feczk001

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src":"/home/miran045/reine097/projects/AlexNet_Abrol2021/reprex"
module load cuda/11.2
module load cuda-sdk/11.2
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/training.py --num-workers=1 --batch-size=1 \
  --tb-prefix="age_regression" --epochs=100 --model="AlexNet3D_Dropout_Regression" "dcan"
