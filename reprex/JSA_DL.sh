#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 2
#SBATCH -p v100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 24:00:00
#SBATCH -J DL
#SBATCH -e ../slogs/err%A-%a.err
#SBATCH -o ../slogs/out%A-%a.out
#SBATCH -A feczk001
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=reiners@umn.edu

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source /home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/activate

./run_DL.sh

sleep 5s
