#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8000
#SBATCH -t 7200
#SBATCH -J DL_Reg
#SBATCH -e slurm_logs/err%A-%a.err
#SBATCH -o slurm_logs/out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abrolanees@gmail.com

sleep 7s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /home/users/aabrol/anaconda3/bin/activate

./run_DL_reg.sh

sleep 8s
