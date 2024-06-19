#!/bin/bash --login

#SBATCH --account yshresth_ai_management
#SBATCH --partition gpu
#SBATCH --output runlog_%A.out
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G

module load gcc/10.4.0 miniconda3/4.10.3
conda activate dl

python main.py