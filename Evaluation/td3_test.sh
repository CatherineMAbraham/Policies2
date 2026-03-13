#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=10      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --time=10:00:00
#SBATCH --output=.out

module load Anaconda3/2024.02-1

source activate softsurg
# Run the script
#srun --export=ALL 
python td3.py --maxforce 3.5 --softtissue "$TISSUE" --num_eps 10 -- model_path '/users/cop21cma/Policies2/TD3/best_models/1/model-03130735-euler-0.001-0.08726646259971647-1'