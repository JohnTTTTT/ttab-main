#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=100000M
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=20jjt@queensu.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs_narval/%x.%j.out
#SBATCH --error=slurm_logs_narval/%x.%j.err
module load python/3.11.4
source myenv/bin/activate

python data_statistics.py