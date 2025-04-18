#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=rrg-amiilab
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
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

python run_exps.py