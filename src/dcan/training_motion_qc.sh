#!/bin/sh

#SBATCH --job-name=motion-training-alex-net # job name

#SBATCH --mem=90g        # memory per cpu-core (what is the default?)
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e motion-alex-net-training-%j.err
#SBATCH -o motion-alex-net-training-%j.out

#SBATCH -A rando149

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/training.py --num-workers=4 --batch-size=8 \
  --tb-prefix="motion_qc_score_regression" --epochs=100 --dset="MRIMotionQcScoreDataset" "dcan"
