#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=rrg-amiilab
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100000M
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=20jjt@queensu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs_narval/%x.%j.out
#SBATCH --error=slurm_logs_narval/%x.%j.err

module load StdEnv/2020
module load gcc/9.3.0
module load python/3.10.2
source /home/johnt/projects/rrg-amiilab/johnt/ttab-main/myenv/bin/activate
module load opencv/4.5.5

# Run your training script using torchrun
#nohup python -u ../pretrain/ssl_pretrain.py --data-name affectnet --model-name resnet50 --batch_size 128 --lr 0.1 --maxEpochs 100 > logs_4gpu.out 2>&1 &
sleep 432000