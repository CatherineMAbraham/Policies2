#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=10     # 4 CPUs per agent
#SBATCH --array=1-2
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --time=02:00:00


module load Anaconda3/2024.02-1
source activate softsurg

# Run the script
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" model_log.csv)
IFS=',' read -r MODEL <<< "$PARAM_LINE"
MODEL=${MODEL//\'/}
echo "Testing model: $MODEL"
#srun --export=ALL 
python env_test2.py --num_eps 10000 --n_envs 10 --model_path "$MODEL" --maxforce 4 --softtissue soft --youngs_modulus 1e7 --log 1