#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=rrg-amiilab
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --mem 100000M
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=20jjt@queensu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs_narval/%x.%j.out
#SBATCH --error=slurm_logs_narval/%x.%j.err
module load python/3.11.4
module load gcc opencv/4.8.1


cd /lustre03/project/6090504/johnt/ttab-main

export PYTHONPATH=$(pwd):$PYTHONPATH

source /home/johnt/projects/rrg-amiilab/johnt/ttab-main/myenv/bin/activate
# pip install --no-index --upgrade pip
# pip install -r requirements.txt

python pretrain/ssl_pretrain.py --data-name affectnet --model-name resnet50 --batch_size 128 --lr 0.01 --maxEpochs 150