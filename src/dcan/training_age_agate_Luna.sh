#!/bin/sh

#SBATCH --job-name=luna.training.agate

#SBATCH --mem=90g
#SBATCH --time=0:30:00
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=6

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e alex-net-training-%j.err
#SBATCH -o alex-net-training-%j.out

#SBATCH -A feczk001

pwd; hostname; date
echo jobid=${SLURM_JOB_ID}; echo nodelist=${SLURM_JOB_NODELIST}

module load python3/3.8.3_anaconda2020.07_mamba
__conda_setup="$(`which conda` 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

cd /home/miran045/reine097/projects/AlexNet_Abrol2021 || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/AlexNet_Abrol2021/src":"/home/miran045/reine097/projects/AlexNet_Abrol2021/reprex"
# source /panfs/roc/msisoft/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh
conda activate /home/miran045/reine097/pytorch_1.11.0_agate-env

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

/home/miran045/reine097/pytorch_1.11.0_agate-env/bin/python \
  /home/miran045/reine097/projects/AlexNet_Abrol2021/src/dcan/training.py --num-workers=1 --batch-size=1 \
  --tb-prefix="age_regression" --epochs=100 --model="luna" "dcan"
