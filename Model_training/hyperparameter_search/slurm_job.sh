#!/bin/bash
#SBATCH -A NAME-GPU
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=my_email
#SBATCH --job-name EM
#SBATCH --output log.%j.log
#SBATCH --gres=gpu:1

source /path/to/python_env/cirkit/bin/activate

date

python3 py_EM.py

date

echo 'bash finish'
