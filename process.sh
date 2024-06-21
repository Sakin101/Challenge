#!/bin/bash --login

#SBATCH --account yshresth_ai_management
#SBATCH --output runlog_%A.out
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 48G

module load gcc/10.4.0 miniconda3/4.10.3
conda activate dl

time python get_videos.py