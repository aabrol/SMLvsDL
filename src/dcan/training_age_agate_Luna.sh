#!/bin/bash -l
#SBATCH --job-name=luna.training.agate
#SBATCH --time=2:00:00
#SBATCH --partition=a100-4
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=luna.training.agate-%j.out
#SBATCH --error=luna.training.agate-%j.err

pwd; hostname; date
echo jobid=${SLURM_JOB_ID}; echo nodelist=${SLURM_JOB_NODELIST}

module load python3/3.8.3_anaconda2020.07_mamba
__conda_setup="$(`which conda` 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

conda activate /panfs/roc/groups/4/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/torch-env

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src":"/home/miran045/reine097/projects/AlexNet_Abrol2021/reprex"
python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/training.py --num-workers=1 --batch-size=1 \
  --tb-prefix="age_regression" --epochs=10 --model="luna" "dcan"

echo COMPLETE
