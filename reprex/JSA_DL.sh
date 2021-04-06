#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 2
#SBATCH -p qTRDGPUL
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 7200
#SBATCH -J DL
#SBATCH -e ../slogs/err%A-%a.err
#SBATCH -o ../slogs/out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aabrol@gsu.edu

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /home/users/aabrol/anaconda3/bin/activate
conda activate AA_DL

./run_DL.sh

sleep 5s
