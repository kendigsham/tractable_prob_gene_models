#!/bin/bash
#SBATCH -A BMAI-CDT-SL2-GPU
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=my_email
#SBATCH --job-name best_model
#SBATCH --output log.%j.log
#SBATCH --gres=gpu:1

source /home/ycks3/python_env/cirkit/bin/activate

date

python3 py_train_best_model_20Feb.py

date

echo 'bash finish'
