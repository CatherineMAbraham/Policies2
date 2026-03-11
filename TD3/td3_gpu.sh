#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --array=1-8
#SBATCH --time=28:00:00
#SBATCH --output=out_%A_%a.out

module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from params_curr_compare.csv
PARAM_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" tests.csv)
IFS=',' read -r TISSUE NUM_SPRINGS CONTACT <<< "$PARAM_LINE"
# Run the script
#srun --export=ALL 
python td3.py --threshold_pos 0.001 --threshold_ori 5 --action_type euler --maxforce 3.5 --softtissue "$TISSUE" --num_springs "$NUM_SPRINGS" --contact_type "$CONTACT" --ran $SLURM_ARRAY_TASK_ID
