#!/bin/sh

#SBATCH --job-name=alex-net.training # job name
#SBATCH --mem=90g        # memory per cpu-core (what is the default?)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e alex-net-training-%j.err
#SBATCH -o alex-net-training-%j.out

#SBATCH -A miran045

rm -f alex-net-training-*.* || true
rm -r /home/miran045/reine097/projects/AlexNet_Abrol2021/data-unversioned/cache
cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/training.py
