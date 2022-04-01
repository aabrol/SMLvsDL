#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 2
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4000
#SBATCH -t 7200
#SBATCH -J DL
#SBATCH -e slurm_logs/err%A-%a.err
#SBATCH -o slurm_logs/out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aabrol@gsu.edu

export SLURM_ARRAY_TASK_ID 0

sleep 7s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /home/users/aabrol/anaconda3/bin/activate

./run_DL.sh

sleep 28s
