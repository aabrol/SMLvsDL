#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRDHM
#SBATCH -c 8
#SBATCH --mem-per-cpu=8000
#SBATCH -t 7200
#SBATCH -J SMLReg
#SBATCH -e slurm_logs/err%A-%a.err
#SBATCH -o slurm_logsAout%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abrolanees@gmail.com

sleep 7s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /home/users/aabrol/anaconda3/bin/activate

python run_SML_reg.py --ml './results_SML_reg_Age/' --scorename 'age' --dataset 'UKBB'
python run_SML_reg.py --ml './results_SML_reg_MMSE/' --scorename 'MMSE' --dataset 'ADNI'

sleep 28s