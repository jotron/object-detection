#!/bin/sh
#SBATCH --account=ie-idi          # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=visual1
#SBATCH --time=0-12:15:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Setting the number of GPUs to 1
#SBATCH --mem=32G                 
#SBATCH --nodes=1
#SBATCH --output=logs/jupyter.txt      # Specifying 'stdout'
#SBATCH --error=logs/jupyter.err        # Specifying 'stderr'
#SBATCH --mail-user=joelra@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1

nvidia-smi
conda activate ./envs
python train.py