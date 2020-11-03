#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4000
#SBATCH -t 7200
#SBATCH -J lasfeo_ar
#SBATCH -e ./slurmlogs/err%A-%a.csv
#SBATCH -o ./slurmlogs/out%A-%a.csv
#SBATCH -A PSYC0002
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aabrol@gsu.edu
#SBATCH --signal=SIGUSR1@90

sleep 7s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate /home/users/aabrol/anaconda3/envs/chumps/
./run_train.sh

sleep 8s
