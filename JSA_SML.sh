#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRDHM
#SBATCH -c 8
#SBATCH --mem-per-cpu=10000
#SBATCH -t 7200
#SBATCH -J SML
#SBATCH -e slurm_logs/err%A-%a.err
#SBATCH -o slurm_logs/out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aabrol@gsu.edu

sleep 7s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /home/users/aabrol/anaconda3/bin/activate

python run_SML.py --mi 10000 --ml './results_SML/'

sleep 28s